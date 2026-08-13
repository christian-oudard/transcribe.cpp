package transcribe

/*
#include <stdlib.h>
#include <transcribe.h>
*/
import "C"

import (
	"errors"
	"fmt"
	"math"
	"time"
	"unsafe"
)

// Model is a loaded GGUF model. It holds the weights, so it is the expensive
// thing to create and the thing worth keeping resident. One model backs any
// number of sessions and must outlive all of them.
type Model struct {
	c *C.struct_transcribe_model
	// borrowed marks a model owned by a session, reached through
	// Session.Model. Freeing one of those would double free.
	borrowed bool
}

// LoadOptions selects where a model's compute runs. The zero value means
// library defaults: probe for the fastest backend, and let it pick the
// device.
type LoadOptions struct {
	Backend Backend
	// GPUDevice picks a device by its index in Devices(). Zero means auto,
	// which is why index 0 is not selectable explicitly: it is reachable
	// only by being first in probe order. Non-zero against a CPU backend
	// request, a non-GPU device, or an out-of-range index is ErrInvalidArg.
	GPUDevice int
}

// LoadModel reads a GGUF model from path. Pass nil opts for defaults.
func LoadModel(path string, opts *LoadOptions) (*Model, error) {
	if err := ready(); err != nil {
		return nil, err
	}
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	p := loadParams(opts)

	m := &Model{}
	if err := check(C.transcribe_model_load_file(cpath, &p, &m.c)); err != nil {
		return nil, err
	}
	return m, nil
}

// loadParams builds C load params, the mirror of sessionParams. Open needs
// the same block, so it lives here rather than inline in LoadModel.
func loadParams(opts *LoadOptions) C.struct_transcribe_model_load_params {
	var p C.struct_transcribe_model_load_params
	C.transcribe_model_load_params_init(&p)
	if opts != nil {
		p.backend = C.transcribe_backend_request(opts.Backend)
		p.gpu_device = C.int(opts.GPUDevice)
	}
	return p
}

// Close frees the model. Every session made from it must already be closed.
// Closing twice is a no-op, and so is closing a model borrowed from a
// session, which that session frees.
func (m *Model) Close() {
	if m.borrowed {
		return
	}
	C.transcribe_model_free(m.c)
	m.c = nil
}

// Arch is the model's architecture slug, e.g. "whisper" or "parakeet".
func (m *Model) Arch() string {
	return C.GoString(C.transcribe_model_arch_string(m.c))
}

// Variant is the specific model within that architecture.
func (m *Model) Variant() string {
	return C.GoString(C.transcribe_model_variant_string(m.c))
}

// Backend is the backend the model actually landed on, which is the answer
// to "did my Auto request reach the GPU?".
func (m *Model) Backend() string {
	return C.GoString(C.transcribe_model_backend(m.c))
}

// Meta reads a scalar string from the GGUF metadata, or "" when the key is
// absent. Numeric hyperparameters and arrays are not exposed.
func (m *Model) Meta(key string) string {
	ckey := C.CString(key)
	defer C.free(unsafe.Pointer(ckey))
	return C.GoString(C.transcribe_model_meta_val_str(m.c, ckey))
}

// Device is the compute device holding this model's weights. MemoryFree on
// the result is a fresh snapshot, so this is also how to ask how much room
// is left on the device a model landed on.
func (m *Model) Device() (Device, error) {
	var cd C.struct_transcribe_backend_device
	C.transcribe_backend_device_init(&cd)
	if err := check(C.transcribe_model_get_device(m.c, &cd)); err != nil {
		return Device{}, err
	}
	return goDevice(&cd), nil
}

// Feature is a behavioral toggle a model may or may not implement. These are
// the properties with no allied data on Capabilities.
type Feature int

const (
	FeatureInitialPrompt       Feature = C.TRANSCRIBE_FEATURE_INITIAL_PROMPT
	FeatureTemperatureFallback Feature = C.TRANSCRIBE_FEATURE_TEMPERATURE_FALLBACK
	FeatureLongForm            Feature = C.TRANSCRIBE_FEATURE_LONG_FORM
	FeatureCancellation        Feature = C.TRANSCRIBE_FEATURE_CANCELLATION
	FeaturePNC                 Feature = C.TRANSCRIBE_FEATURE_PNC
	FeatureITN                 Feature = C.TRANSCRIBE_FEATURE_ITN
	FeatureDiarization         Feature = C.TRANSCRIBE_FEATURE_DIARIZATION
)

// Supports probes a feature. Setting a run option a model does not support
// is usually a warning rather than an error, so probe first if the
// difference matters.
func (m *Model) Supports(f Feature) bool {
	return bool(C.transcribe_model_supports(m.c, C.transcribe_feature(f)))
}

// Capabilities are semantic properties read from the GGUF at load time. They
// never change with backend fallback.
type Capabilities struct {
	// NativeSampleRate is what the model was trained on. The API still
	// takes 16 kHz PCM whatever this says.
	NativeSampleRate int
	// Languages are the codes accepted as a language hint. Empty means
	// the model does not advertise a set, not that it accepts none.
	Languages []string
	// MaxTimestamps is the finest granularity the model can produce.
	// Asking for finer is ErrUnsupportedStamps.
	MaxTimestamps Timestamps

	SupportsLanguageDetect bool
	SupportsTranslate      bool
	SupportsStreaming      bool
	SupportsSpecDecode     bool

	// MaxAudio is the longest clip one Run accepts, or 0 for no practical
	// limit because the family chunks internally. A session that lowers
	// Context lowers the real limit below this; read that one from
	// Session.Limits.
	MaxAudio time.Duration
	// TranslateTargets narrows which target languages a translate task
	// accepts. Empty means unadvertised rather than none.
	TranslateTargets []string
}

// Capabilities reads the model's capability block.
func (m *Model) Capabilities() (Capabilities, error) {
	var c C.struct_transcribe_capabilities
	C.transcribe_capabilities_init(&c)
	if err := check(C.transcribe_model_get_capabilities(m.c, &c)); err != nil {
		return Capabilities{}, err
	}
	return Capabilities{
		NativeSampleRate:       int(c.native_sample_rate),
		Languages:              goStrings(c.languages, int(c.n_languages)),
		MaxTimestamps:          stampsFromC(c.max_timestamp_kind),
		SupportsLanguageDetect: bool(c.supports_language_detect),
		SupportsTranslate:      bool(c.supports_translate),
		SupportsStreaming:      bool(c.supports_streaming),
		SupportsSpecDecode:     bool(c.supports_spec_decode),
		MaxAudio:               ms(c.max_audio_ms),
		TranslateTargets:       goStrings(c.translate_target_languages, int(c.n_translate_target_languages)),
	}, nil
}

// ErrNoTokenizer is what Tokenize returns for a model whose vocabulary has
// no encode path wired up, which today is every SentencePiece family.
var ErrNoTokenizer = errors.New("model cannot tokenize text")

// Tokenize encodes plain UTF-8 text into the model's vocabulary, without any
// BOS, EOS or <|...|> markers. Special tokens in the input are not
// recognized and get encoded piece by piece, so pass only plain text.
func (m *Model) Tokenize(text string) ([]int32, error) {
	ctext := C.CString(text)
	defer C.free(unsafe.Pointer(ctext))

	// The library reports a short buffer as the negative of what it needs,
	// so ask with none and size the real call from the answer.
	n := int(C.transcribe_tokenize(m.c, ctext, nil, 0))
	switch {
	case n == math.MinInt32:
		return nil, ErrNoTokenizer
	case n == 0:
		return nil, nil
	case n > 0:
		// Nothing fits in a zero-length buffer, so a non-negative
		// count here would mean the library ignored n_max.
		return nil, fmt.Errorf("transcribe_tokenize sized %d tokens into no buffer", n)
	}

	out := make([]int32, -n)
	got := int(C.transcribe_tokenize(m.c, ctext, (*C.int32_t)(&out[0]), C.size_t(len(out))))
	if got < 0 {
		return nil, fmt.Errorf("transcribe_tokenize wanted %d tokens after asking for %d", -got, len(out))
	}
	return out[:got], nil
}

// goStrings copies a C array of n string pointers into Go strings.
func goStrings(arr **C.char, n int) []string {
	if arr == nil || n <= 0 {
		return nil
	}
	slice := unsafe.Slice(arr, n)
	out := make([]string, n)
	for i, s := range slice {
		out[i] = C.GoString(s)
	}
	return out
}
