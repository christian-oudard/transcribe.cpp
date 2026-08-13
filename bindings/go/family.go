package transcribe

// Family extensions: the per-family knobs that do not generalize across
// architectures, reached through RunOptions.Family and StreamOptions.Family.
//
// This file includes the umbrella header rather than transcribe.h, which is
// what pulls in every family header the install ships.

/*
#include <stdlib.h>
#include <transcribe/extensions.h>
*/
import "C"

import "unsafe"

// Opt wraps a value for the optional fields below, where a nil pointer means
// the family's own default and a set one overrides it:
//
//	&transcribe.WhisperRunOptions{Temperature: transcribe.Opt[float32](0.2)}
//
// The families disagree about which values are sentinels and several of them
// take a meaningful zero, so a pointer is the only encoding that separates
// "leave it alone" from "set it to zero" for every field.
func Opt[T any](v T) *T { return &v }

// ExtSlot is the API surface an extension is passed to. A kind is legal in
// exactly one slot, which the Go types enforce at compile time: a stream
// extension does not satisfy RunExtension.
type ExtSlot int

const (
	SlotRun    ExtSlot = C.TRANSCRIBE_EXT_SLOT_RUN
	SlotStream ExtSlot = C.TRANSCRIBE_EXT_SLOT_STREAM
)

// ExtKind identifies one family extension's schema. See
// docs/extension-kinds.md for the registry.
type ExtKind uint32

const (
	KindWhisperRun             ExtKind = C.TRANSCRIBE_EXT_KIND_WHISPER_RUN
	KindVoxtralRun             ExtKind = C.TRANSCRIBE_EXT_KIND_VOXTRAL_RUN
	KindSortformerStream       ExtKind = C.TRANSCRIBE_EXT_KIND_SORTFORMER_STREAM
	KindParakeetStream         ExtKind = C.TRANSCRIBE_EXT_KIND_PARAKEET_STREAM
	KindParakeetBufferedStream ExtKind = C.TRANSCRIBE_EXT_KIND_PARAKEET_BUFFERED_STREAM
	KindMoonshineStreaming     ExtKind = C.TRANSCRIBE_EXT_KIND_MOONSHINE_STREAMING_STREAM
	KindVoxtralRealtimeStream  ExtKind = C.TRANSCRIBE_EXT_KIND_VOXTRAL_REALTIME_STREAM
)

// AcceptsExtension reports whether this model variant takes the given
// extension kind in the given slot. Acceptance is per variant, not per
// family: parakeet ships one streaming schema for its cache-aware variants
// and another for its chunked-attention ones, and each rejects the other.
//
// Passing an extension a model does not accept is ErrInvalidArg from the run
// or the stream begin, so probe when the model is not known up front.
func (m *Model) AcceptsExtension(slot ExtSlot, kind ExtKind) bool {
	return bool(C.transcribe_model_accepts_ext_kind(m.c, C.transcribe_ext_slot(slot), C.uint32_t(kind)))
}

// RunExtension is a family's own block of run options. Only whisper and
// sortformer have one; the method is unexported, so the set is closed.
type RunExtension interface {
	Kind() ExtKind
	// runExt builds the typed struct in C memory and returns it with the
	// function that releases it. C memory rather than Go: cgo will not
	// take a pointer to Go memory that itself holds Go pointers, which the
	// params struct would once its family field is set.
	runExt() (*C.struct_transcribe_ext, func())
}

// StreamExtension is a family's own block of streaming options.
type StreamExtension interface {
	Kind() ExtKind
	streamExt() (*C.struct_transcribe_ext, func())
}

// alloc reserves n bytes of C memory and returns them with their free.
func alloc(n uintptr) (unsafe.Pointer, func()) {
	p := C.malloc(C.size_t(n))
	return p, func() { C.free(p) }
}

// WhisperPromptCondition is how far an initial prompt reaches.
type WhisperPromptCondition int

const (
	// PromptFirstSegment conditions only the first 30-second window, which
	// is whisper's own recipe and the default.
	PromptFirstSegment WhisperPromptCondition = C.TRANSCRIBE_WHISPER_PROMPT_FIRST_SEGMENT
	// PromptAllSegments repeats the prompt for every window.
	PromptAllSegments WhisperPromptCondition = C.TRANSCRIBE_WHISPER_PROMPT_ALL_SEGMENTS
)

// WhisperRunOptions are whisper's run knobs: the initial prompt, the
// temperature-fallback ladder and the thresholds that drive it. A nil field
// keeps whisper's own shipping recipe, which is not the same as the zero
// value, so set only what you mean to change.
type WhisperRunOptions struct {
	// InitialPrompt is prior context as text, tokenized the way HF's
	// get_prompt_ids does. A special tag in it is ErrInvalidArg.
	InitialPrompt string
	// PromptTokens is the same thing already tokenized, without the
	// startofprev marker, which the library prepends. It wins over
	// InitialPrompt when both are set.
	PromptTokens []int32
	// PromptCondition defaults to PromptFirstSegment. Setting it to
	// PromptAllSegments without also setting ConditionOnPrevTokens is
	// ErrInvalidArg: HF ties the two together, and the library keeps that
	// parity rather than silently picking one.
	PromptCondition *WhisperPromptCondition
	// ConditionOnPrevTokens carries the tail of the previous window's
	// tokens into the next one.
	ConditionOnPrevTokens *bool
	// MaxPrevContextTokens caps what ConditionOnPrevTokens carries.
	MaxPrevContextTokens *int

	// Temperature is the first tier, TemperatureInc the step the fallback
	// ladder climbs by when a tier is rejected.
	Temperature    *float32
	TemperatureInc *float32
	// CompressionRatioThreshold, LogProbThreshold and NoSpeechThreshold
	// are the metrics that reject a tier.
	CompressionRatioThreshold *float32
	LogProbThreshold          *float32
	NoSpeechThreshold         *float32

	// Seed makes sampling reproducible above temperature 0, where the
	// decode is nondeterministic. Ignored at temperature 0.
	Seed *uint32
	// MaxInitialTimestamp caps the first timestamp, in seconds.
	MaxInitialTimestamp *float32
}

func (o *WhisperRunOptions) Kind() ExtKind { return KindWhisperRun }

func (o *WhisperRunOptions) runExt() (*C.struct_transcribe_ext, func()) {
	mem, free := alloc(unsafe.Sizeof(C.struct_transcribe_whisper_run_ext{}))
	e := (*C.struct_transcribe_whisper_run_ext)(mem)
	C.transcribe_whisper_run_ext_init(e)

	frees := []func(){free}
	if o.InitialPrompt != "" {
		c := C.CString(o.InitialPrompt)
		e.initial_prompt = c
		frees = append(frees, func() { C.free(unsafe.Pointer(c)) })
	}
	if len(o.PromptTokens) > 0 {
		toks, freeToks := alloc(uintptr(len(o.PromptTokens)) * unsafe.Sizeof(C.int32_t(0)))
		copy(unsafe.Slice((*int32)(toks), len(o.PromptTokens)), o.PromptTokens)
		e.prompt_tokens = (*C.int32_t)(toks)
		e.n_prompt_tokens = C.size_t(len(o.PromptTokens))
		frees = append(frees, freeToks)
	}
	if o.PromptCondition != nil {
		e.prompt_condition = C.enum_transcribe_whisper_prompt_condition(*o.PromptCondition)
	}
	if o.ConditionOnPrevTokens != nil {
		e.condition_on_prev_tokens = C.bool(*o.ConditionOnPrevTokens)
	}
	if o.MaxPrevContextTokens != nil {
		e.max_prev_context_tokens = C.int32_t(*o.MaxPrevContextTokens)
	}
	if o.Temperature != nil {
		e.temperature = C.float(*o.Temperature)
	}
	if o.TemperatureInc != nil {
		e.temperature_inc = C.float(*o.TemperatureInc)
	}
	if o.CompressionRatioThreshold != nil {
		e.compression_ratio_thold = C.float(*o.CompressionRatioThreshold)
	}
	if o.LogProbThreshold != nil {
		e.logprob_thold = C.float(*o.LogProbThreshold)
	}
	if o.NoSpeechThreshold != nil {
		e.no_speech_thold = C.float(*o.NoSpeechThreshold)
	}
	if o.Seed != nil {
		e.seed = C.uint32_t(*o.Seed)
	}
	if o.MaxInitialTimestamp != nil {
		e.max_initial_timestamp = C.float(*o.MaxInitialTimestamp)
	}
	return (*C.struct_transcribe_ext)(mem), func() {
		for _, f := range frees {
			f()
		}
	}
}

// VoxtralRunOptions is voxtral's free-text instruction, which is what the
// family's initial-prompt capability actually is.
//
// Voxtral is an audio-LLM, so this is an instruction to a language model
// rather than whisper's decoder conditioning: it lands after the audio tokens
// inside the instruct template, and the model is free to follow it loosely or
// not at all. Biasing a transcript toward known vocabulary is the use it was
// exposed for, e.g. "Transcribe. Expected terms: NixOS, nixpkgs, direnv."
//
// It shares the decoder's context window with the audio and the transcript,
// which voxtral caps hard, so keep it short. Combining it with TaskTranslate
// is ErrInvalidArg, since both want the one instruction slot.
type VoxtralRunOptions struct {
	Instruction string
}

func (o *VoxtralRunOptions) Kind() ExtKind { return KindVoxtralRun }

func (o *VoxtralRunOptions) runExt() (*C.struct_transcribe_ext, func()) {
	mem, free := alloc(unsafe.Sizeof(C.struct_transcribe_voxtral_run_ext{}))
	e := (*C.struct_transcribe_voxtral_run_ext)(mem)
	C.transcribe_voxtral_run_ext_init(e)
	if o.Instruction == "" {
		return (*C.struct_transcribe_ext)(mem), free
	}
	c := C.CString(o.Instruction)
	e.instruction = c
	return (*C.struct_transcribe_ext)(mem), func() {
		C.free(unsafe.Pointer(c))
		free()
	}
}

// SortformerPreset is a jointly tuned latency and accuracy operating point,
// not a dial: the values in between are not meaningful.
type SortformerPreset int

const (
	// SortformerDefault keeps whatever the GGUF shipped with.
	SortformerDefault SortformerPreset = C.TRANSCRIBE_SORTFORMER_PRESET_DEFAULT
	// SortformerVeryHighLatency is ~30.4s of lookahead, the published
	// operating point for offline files and the most accurate.
	SortformerVeryHighLatency SortformerPreset = C.TRANSCRIBE_SORTFORMER_PRESET_VERY_HIGH_LATENCY
	// SortformerHighLatency is ~10.0s of lookahead.
	SortformerHighLatency SortformerPreset = C.TRANSCRIBE_SORTFORMER_PRESET_HIGH_LATENCY
	// SortformerLowLatency is ~1.04s of lookahead, the real-time point. It
	// costs considerably more compute per second of audio than the larger
	// chunks do.
	SortformerLowLatency SortformerPreset = C.TRANSCRIBE_SORTFORMER_PRESET_LOW_LATENCY
)

// SortformerStreamOptions is sortformer's operating point. Despite the name
// it rides the run slot, since sortformer streams inside one run.
type SortformerStreamOptions struct {
	Preset *SortformerPreset
}

func (o *SortformerStreamOptions) Kind() ExtKind { return KindSortformerStream }

func (o *SortformerStreamOptions) runExt() (*C.struct_transcribe_ext, func()) {
	mem, free := alloc(unsafe.Sizeof(C.struct_transcribe_sortformer_stream_ext{}))
	e := (*C.struct_transcribe_sortformer_stream_ext)(mem)
	C.transcribe_sortformer_stream_ext_init(e)
	if o.Preset != nil {
		e.preset = C.transcribe_sortformer_preset(*o.Preset)
	}
	return (*C.struct_transcribe_ext)(mem), free
}

// ParakeetStreamOptions are the cache-aware parakeet streaming knobs.
type ParakeetStreamOptions struct {
	// AttContextRight is how many frames of right context attention may
	// see, which trades latency against accuracy.
	AttContextRight *int
}

func (o *ParakeetStreamOptions) Kind() ExtKind { return KindParakeetStream }

func (o *ParakeetStreamOptions) streamExt() (*C.struct_transcribe_ext, func()) {
	mem, free := alloc(unsafe.Sizeof(C.struct_transcribe_parakeet_stream_ext{}))
	e := (*C.struct_transcribe_parakeet_stream_ext)(mem)
	C.transcribe_parakeet_stream_ext_init(e)
	if o.AttContextRight != nil {
		e.att_context_right = C.int32_t(*o.AttContextRight)
	}
	return (*C.struct_transcribe_ext)(mem), free
}

// ParakeetBufferedStreamOptions are the chunked-attention parakeet knobs,
// the schema the unified variants take instead of ParakeetStreamOptions.
type ParakeetBufferedStreamOptions struct {
	// Left and Right are the context around each Chunk, in milliseconds.
	LeftMS  *int
	ChunkMS *int
	RightMS *int
}

func (o *ParakeetBufferedStreamOptions) Kind() ExtKind { return KindParakeetBufferedStream }

func (o *ParakeetBufferedStreamOptions) streamExt() (*C.struct_transcribe_ext, func()) {
	mem, free := alloc(unsafe.Sizeof(C.struct_transcribe_parakeet_buffered_stream_ext{}))
	e := (*C.struct_transcribe_parakeet_buffered_stream_ext)(mem)
	C.transcribe_parakeet_buffered_stream_ext_init(e)
	if o.LeftMS != nil {
		e.left_ms = C.int32_t(*o.LeftMS)
	}
	if o.ChunkMS != nil {
		e.chunk_ms = C.int32_t(*o.ChunkMS)
	}
	if o.RightMS != nil {
		e.right_ms = C.int32_t(*o.RightMS)
	}
	return (*C.struct_transcribe_ext)(mem), free
}

// MoonshineStreamingOptions are the moonshine streaming knobs.
type MoonshineStreamingOptions struct {
	// MinDecodeIntervalMS floors how often the model re-decodes, so a fast
	// feed does not spend the whole budget on hypotheses nobody sees.
	MinDecodeIntervalMS *int
}

func (o *MoonshineStreamingOptions) Kind() ExtKind { return KindMoonshineStreaming }

func (o *MoonshineStreamingOptions) streamExt() (*C.struct_transcribe_ext, func()) {
	mem, free := alloc(unsafe.Sizeof(C.struct_transcribe_moonshine_streaming_stream_ext{}))
	e := (*C.struct_transcribe_moonshine_streaming_stream_ext)(mem)
	C.transcribe_moonshine_streaming_stream_ext_init(e)
	if o.MinDecodeIntervalMS != nil {
		e.min_decode_interval_ms = C.int32_t(*o.MinDecodeIntervalMS)
	}
	return (*C.struct_transcribe_ext)(mem), free
}

// VoxtralRealtimeStreamOptions are the voxtral-realtime streaming knobs.
type VoxtralRealtimeStreamOptions struct {
	// NumDelayTokens is how far the model lags the audio, its latency and
	// accuracy trade.
	NumDelayTokens *int
	// MinDecodeIntervalMS floors the re-decode rate, as for moonshine.
	MinDecodeIntervalMS *int
}

func (o *VoxtralRealtimeStreamOptions) Kind() ExtKind { return KindVoxtralRealtimeStream }

func (o *VoxtralRealtimeStreamOptions) streamExt() (*C.struct_transcribe_ext, func()) {
	mem, free := alloc(unsafe.Sizeof(C.struct_transcribe_voxtral_realtime_stream_ext{}))
	e := (*C.struct_transcribe_voxtral_realtime_stream_ext)(mem)
	C.transcribe_voxtral_realtime_stream_ext_init(e)
	if o.NumDelayTokens != nil {
		e.num_delay_tokens = C.int32_t(*o.NumDelayTokens)
	}
	if o.MinDecodeIntervalMS != nil {
		e.min_decode_interval_ms = C.int32_t(*o.MinDecodeIntervalMS)
	}
	return (*C.struct_transcribe_ext)(mem), free
}
