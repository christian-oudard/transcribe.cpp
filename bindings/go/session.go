package transcribe

/*
#include <stdlib.h>
#include <transcribe.h>

// Defined in callback.go. Each file has its own preamble, so the
// declaration has to be repeated wherever the trampoline is referenced.
extern bool transcribeAbortTrampoline(void * user_data);
*/
import "C"

import (
	"context"
	"errors"
	"runtime"
	"runtime/cgo"
	"time"
	"unsafe"
)

// SampleRate is the only input rate the library accepts. There is no
// resampler linked in, so callers convert their audio before Run.
const SampleRate = 16000

// Session is one transcription context bound to a model. It holds the
// decoder state and the last result, so it is single-threaded: never call
// two of its methods concurrently. Nor may two sessions of the same model
// compute at once, which is a limitation of the library rather than of this
// binding; load one Model per worker to transcribe in parallel.
type Session struct {
	c *C.struct_transcribe_session
	// stopWatch removes an installed abort callback. A one-shot run takes
	// its down before returning; a stream's has to outlive the call that
	// set it up, so it lives here until finalize or reset.
	stopWatch func()
}

// unwatch takes down any installed abort callback.
func (s *Session) unwatch() {
	if s.stopWatch != nil {
		s.stopWatch()
		s.stopWatch = nil
	}
}

// SessionOptions tunes a session's compute and memory. The zero value means
// library defaults.
type SessionOptions struct {
	// Threads is CPU threads for ops that run on CPU. 0 lets the library
	// choose.
	Threads int
	// KVType is the precision of K/V activations in flash attention.
	KVType KVType
	// Context caps the decoder's context window in tokens, to bound KV
	// cache memory. 0 uses the model's true maximum, which is right for
	// almost everyone; a value above the maximum is clamped down to it.
	// Lowering it can lower the longest clip the session accepts.
	Context int
}

// KVType is the precision of the K/V cache in flash attention.
type KVType int

const (
	// KVAuto is f16 for quantized weights, f32 for f32 weights.
	KVAuto KVType = C.TRANSCRIBE_KV_TYPE_AUTO
	KVF32  KVType = C.TRANSCRIBE_KV_TYPE_F32
	KVF16  KVType = C.TRANSCRIBE_KV_TYPE_F16
)

// NewSession opens a session against an already-loaded model. The model must
// outlive it. Use this when several sessions share one model; use Open for
// the single-stream case.
func (m *Model) NewSession(opts *SessionOptions) (*Session, error) {
	s := &Session{}
	p := sessionParams(opts)
	if err := check(C.transcribe_session_init(m.c, &p, &s.c)); err != nil {
		return nil, err
	}
	return s, nil
}

// Open loads a model and opens one session against it, for the common case
// of transcribing a single stream. The session owns the model, so Close
// frees both. Reach past this only to share one model across sessions.
func Open(path string, load *LoadOptions, sess *SessionOptions) (*Session, error) {
	if err := ready(); err != nil {
		return nil, err
	}
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	lp := loadParams(load)
	sp := sessionParams(sess)

	s := &Session{}
	if err := check(C.transcribe_open(cpath, &lp, &sp, &s.c)); err != nil {
		return nil, err
	}
	return s, nil
}

func sessionParams(opts *SessionOptions) C.struct_transcribe_session_params {
	var p C.struct_transcribe_session_params
	C.transcribe_session_params_init(&p)
	if opts != nil {
		p.n_threads = C.int(opts.Threads)
		p.kv_type = C.transcribe_kv_type(opts.KVType)
		p.n_ctx = C.int32_t(opts.Context)
	}
	return p
}

// Close frees the session, and the model too when Open created it. Results
// read from it are invalid afterwards. Closing twice is a no-op.
func (s *Session) Close() {
	s.unwatch()
	C.transcribe_session_free(s.c)
	s.c = nil
}

// Model borrows the model behind the session, which is how a session from
// Open reaches model introspection. The session owns it, so Close on the
// returned model is a no-op rather than a double free.
func (s *Session) Model() *Model {
	return &Model{c: C.transcribe_get_model(s.c), borrowed: true}
}

// Task is what to do with the audio.
type Task int

const (
	TaskTranscribe Task = C.TRANSCRIBE_TASK_TRANSCRIBE
	// TaskTranslate requires Capabilities.SupportsTranslate; otherwise the
	// run fails with ErrUnsupportedTask.
	TaskTranslate Task = C.TRANSCRIBE_TASK_TRANSLATE
)

// Timestamps is the requested alignment granularity. Anything other than
// Auto is a ceiling: coarser than the model's maximum elides the finer data,
// finer is ErrUnsupportedStamps.
//
// These are not the C enum's values. Go's zero value has to be the library
// default, which is Auto, so the two are mapped rather than shared.
type Timestamps int

const (
	// StampsAuto asks for the richest the model can produce and is never
	// rejected.
	StampsAuto Timestamps = iota
	StampsNone
	StampsSegment
	StampsWord
	StampsToken
)

var stampsToC = [...]C.transcribe_timestamp_kind{
	StampsAuto:    C.TRANSCRIBE_TIMESTAMPS_AUTO,
	StampsNone:    C.TRANSCRIBE_TIMESTAMPS_NONE,
	StampsSegment: C.TRANSCRIBE_TIMESTAMPS_SEGMENT,
	StampsWord:    C.TRANSCRIBE_TIMESTAMPS_WORD,
	StampsToken:   C.TRANSCRIBE_TIMESTAMPS_TOKEN,
}

func stampsFromC(k C.transcribe_timestamp_kind) Timestamps {
	for i, c := range stampsToC {
		if c == k {
			return Timestamps(i)
		}
	}
	return StampsAuto
}

func (t Timestamps) String() string {
	switch t {
	case StampsAuto:
		return "auto"
	case StampsNone:
		return "none"
	case StampsSegment:
		return "segment"
	case StampsWord:
		return "word"
	case StampsToken:
		return "token"
	}
	return "unknown"
}

// Mode is a three-state runtime toggle: leave it at the family's default, or
// force it on or off. Forcing it on a model that does not implement the
// toggle logs a warning and proceeds with the default.
type Mode int

const (
	ModeDefault Mode = iota
	ModeOff
	ModeOn
)

// One Go type covers three separate C enums, so each gets its own table
// rather than a cast that assumes the three keep agreeing.
var (
	pncToC = [...]C.enum_transcribe_pnc_mode{
		ModeDefault: C.TRANSCRIBE_PNC_MODE_DEFAULT,
		ModeOff:     C.TRANSCRIBE_PNC_MODE_OFF,
		ModeOn:      C.TRANSCRIBE_PNC_MODE_ON,
	}
	itnToC = [...]C.enum_transcribe_itn_mode{
		ModeDefault: C.TRANSCRIBE_ITN_MODE_DEFAULT,
		ModeOff:     C.TRANSCRIBE_ITN_MODE_OFF,
		ModeOn:      C.TRANSCRIBE_ITN_MODE_ON,
	}
	diarizeToC = [...]C.enum_transcribe_diarize_mode{
		ModeDefault: C.TRANSCRIBE_DIARIZE_MODE_DEFAULT,
		ModeOff:     C.TRANSCRIBE_DIARIZE_MODE_OFF,
		ModeOn:      C.TRANSCRIBE_DIARIZE_MODE_ON,
	}
)

// RunOptions are per-run settings. The zero value is library defaults:
// transcribe, automatic timestamps, family-default punctuation, and
// autodetected language where the model can do that.
type RunOptions struct {
	Task       Task
	Timestamps Timestamps
	// PNC toggles punctuation and capitalization, ITN toggles inverse text
	// normalization ("twenty five" to "25"), Diarize toggles speaker
	// attribution. Each needs the matching Feature.
	PNC     Mode
	ITN     Mode
	Diarize Mode
	// Language is a source-language hint like "en", or empty to
	// autodetect, which needs Capabilities.SupportsLanguageDetect.
	Language string
	// TargetLanguage is the language to translate into, for TaskTranslate.
	TargetLanguage string
	// KeepSpecialTags leaves control tags such as <|...|> inline in the
	// text. Result.RawText gives the uncleaned decode for every family
	// without giving up the clean Text, and is usually what you want.
	KeepSpecialTags bool
	// SpecDrafts is how many tokens speculative decoding drafts per verify
	// pass: 0 for the family's tuned default, SpecDecodeOff to disable, or
	// a count, in practice 1 to 8. The best count is hardware-dependent.
	// Ignored unless Capabilities.SupportsSpecDecode.
	SpecDrafts int
	// Family carries knobs that only one architecture has, such as
	// WhisperRunOptions. A model that does not accept the kind fails the
	// run with ErrInvalidArg; Model.AcceptsExtension probes for it.
	Family RunExtension
}

// SpecDecodeOff disables speculative decoding, for reproducing pre-spec
// output or measuring a baseline. RunOptions.SpecDrafts left at zero means
// the family default instead.
const SpecDecodeOff = -1

// runParams builds C run params. Its returned free must be called once the
// run returns, since the library copies the strings it needs before then.
func runParams(opts *RunOptions) (C.struct_transcribe_run_params, func()) {
	var p C.struct_transcribe_run_params
	C.transcribe_run_params_init(&p)
	if opts == nil {
		return p, func() {}
	}
	p.task = C.transcribe_task(opts.Task)
	p.timestamps = stampsToC[opts.Timestamps]
	p.pnc = pncToC[opts.PNC]
	p.itn = itnToC[opts.ITN]
	p.diarize = diarizeToC[opts.Diarize]
	p.keep_special_tags = C.bool(opts.KeepSpecialTags)
	// The C convention is inverted from ours: it spells the family default
	// -1 and "disabled" 0, so that a zeroed struct disables. Ours has to
	// leave the default in the zero value.
	switch opts.SpecDrafts {
	case 0:
		p.spec_k_drafts = -1
	case SpecDecodeOff:
		p.spec_k_drafts = 0
	default:
		p.spec_k_drafts = C.int32_t(opts.SpecDrafts)
	}

	var frees []func()
	cstr := func(s string) *C.char {
		if s == "" {
			return nil
		}
		c := C.CString(s)
		frees = append(frees, func() { C.free(unsafe.Pointer(c)) })
		return c
	}
	p.language = cstr(opts.Language)
	p.target_language = cstr(opts.TargetLanguage)
	if opts.Family != nil {
		ext, free := opts.Family.runExt()
		p.family = ext
		frees = append(frees, free)
	}
	return p, func() {
		for _, f := range frees {
			f()
		}
	}
}

// Run transcribes one utterance of mono float32 PCM at SampleRate, in
// [-1, 1]. The returned Result owns its contents, so it stays readable after
// the next run.
//
// Cancelling ctx aborts the run between decode steps, typically within tens
// of milliseconds, and returns ErrAborted; the result then holds whatever
// completed before the abort. Mid-encoder abort is not supported, so a run
// dominated by its encoder will not stop early.
func (s *Session) Run(ctx context.Context, pcm []float32, opts *RunOptions) (Result, error) {
	if len(pcm) == 0 || s.StreamState() == StreamActive {
		return Result{}, ErrInvalidArg
	}
	p, free := runParams(opts)
	defer free()
	defer s.watch(ctx)()

	err := check(C.transcribe_run(s.c, (*C.float)(&pcm[0]), C.int(len(pcm)), &p))
	runtime.KeepAlive(pcm)
	if err != nil && !partial(err) {
		return Result{}, err
	}
	return reader{s: s, i: -1}.materialize(), err
}

// partial reports whether a failing status still left a usable result behind.
// A run that stopped early wrote what it had; one that failed before running
// may not have touched the session's result storage at all, where the
// previous run's transcript is still sitting.
func partial(err error) bool {
	return errors.Is(err, ErrAborted) || errors.Is(err, ErrOutputTruncated)
}

// BatchResult is one utterance's outcome from RunBatch. Err is that
// utterance's own status: a batch can succeed as a whole with one utterance
// failing inside it, and an utterance that was aborted or truncated carries
// both an error and the partial transcript it managed.
type BatchResult struct {
	Result
	Err error
}

// RunBatch transcribes several utterances in one call. Families with a
// batched compute path dispatch them together, which can roughly double
// throughput on a GPU that one utterance underuses; the rest fall back to
// running them in turn, so every model accepts this.
//
// The returned error is the whole-batch verdict, and results are returned
// with it whenever the batch got far enough to fill them, which is the case
// for a cancelled batch. Per-utterance failures do not show up there at all,
// only in each BatchResult.Err, so check those too.
func (s *Session) RunBatch(ctx context.Context, pcm [][]float32, opts *RunOptions) ([]BatchResult, error) {
	if len(pcm) == 0 || s.StreamState() == StreamActive {
		return nil, ErrInvalidArg
	}
	p, free := runParams(opts)
	defer free()
	defer s.watch(ctx)()

	// The library wants an array of pointers plus an array of lengths, so
	// they go in C memory: a Go slice of Go pointers may not cross into C.
	// The utterances themselves stay where they are, but a Go pointer
	// written into C memory has to be pinned first, since nothing there is
	// visible to the garbage collector.
	var pin runtime.Pinner
	defer pin.Unpin()
	ptrs := C.malloc(C.size_t(len(pcm)) * C.size_t(unsafe.Sizeof(uintptr(0))))
	defer C.free(ptrs)
	lens := C.malloc(C.size_t(len(pcm)) * C.size_t(unsafe.Sizeof(C.int(0))))
	defer C.free(lens)
	ptrSlice := unsafe.Slice((**C.float)(ptrs), len(pcm))
	lenSlice := unsafe.Slice((*C.int)(lens), len(pcm))
	for i, u := range pcm {
		if len(u) == 0 {
			// A zero-length utterance is a per-utterance failure, which
			// the library reports through Result.Err.
			ptrSlice[i], lenSlice[i] = nil, 0
			continue
		}
		pin.Pin(&u[0])
		ptrSlice[i], lenSlice[i] = (*C.float)(&u[0]), C.int(len(u))
	}

	// The pinner keeps every utterance alive and unmoved until Unpin, so
	// there is no KeepAlive to add on top of it.
	err := check(C.transcribe_run_batch(s.c, (**C.float)(ptrs), (*C.int)(lens), C.int(len(pcm)), &p))
	// OK and ErrAborted both leave one filled slot per utterance; any other
	// status means the batch never ran, so there is nothing to read.
	if err != nil && !partial(err) {
		return nil, err
	}

	n := int(C.transcribe_batch_n_results(s.c))
	out := make([]BatchResult, n)
	for i := range out {
		r := reader{s: s, i: i}
		out[i] = BatchResult{Err: r.status()}
		if out[i].Err == nil || partial(out[i].Err) {
			out[i].Result = r.materialize()
		}
	}
	return out, err
}

// watch installs an abort callback driven by ctx for the length of a run,
// and returns the function that removes it again. A context that can never
// be cancelled installs nothing.
func (s *Session) watch(ctx context.Context) func() {
	if ctx == nil || ctx.Done() == nil {
		return func() {}
	}
	// The callback's user_data is a void *, and a cgo.Handle is an integer,
	// so it travels in a C-allocated cell rather than being cast to a
	// pointer it is not.
	h := cgo.NewHandle(ctx)
	cell := C.malloc(C.size_t(unsafe.Sizeof(C.uintptr_t(0))))
	*(*C.uintptr_t)(cell) = C.uintptr_t(h)

	C.transcribe_set_abort_callback(s.c, C.transcribe_abort_callback(C.transcribeAbortTrampoline), cell)
	return func() {
		C.transcribe_set_abort_callback(s.c, nil, nil)
		C.free(cell)
		h.Delete()
	}
}

// Aborted reports whether the last run stopped early because ctx was
// cancelled, which is how partial-from-abort is told from complete.
func (s *Session) Aborted() bool { return bool(C.transcribe_was_aborted(s.c)) }

// Truncated reports whether the last run hit the model's output budget and
// stopped mid-transcript.
func (s *Session) Truncated() bool { return bool(C.transcribe_was_truncated(s.c)) }

// Limits are a session's effective input bounds, which take its Context
// setting into account where the model-level Capabilities do not.
type Limits struct {
	// MaxAudio is the longest clip this session accepts, or 0 for no
	// practical limit.
	MaxAudio time.Duration
}

// Limits reads the session's effective bounds.
func (s *Session) Limits() (Limits, error) {
	var l C.struct_transcribe_session_limits
	C.transcribe_session_limits_init(&l)
	if err := check(C.transcribe_session_get_limits(s.c, &l)); err != nil {
		return Limits{}, err
	}
	return Limits{MaxAudio: ms(l.effective_max_audio_ms)}, nil
}

// ResetTimings zeroes the session's accumulated timings.
func (s *Session) ResetTimings() { C.transcribe_reset_timings(s.c) }
