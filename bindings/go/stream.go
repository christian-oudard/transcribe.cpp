package transcribe

/*
#include <transcribe.h>
*/
import "C"

import (
	"context"
	"time"
)

// StreamState is where a stream is in its lifecycle. A session starts Idle,
// StreamBegin moves it to Active, StreamFinalize to Finished, and a terminal
// error to Failed. StreamReset returns it to Idle from anywhere.
type StreamState int

const (
	StreamIdle     StreamState = C.TRANSCRIBE_STREAM_IDLE
	StreamActive   StreamState = C.TRANSCRIBE_STREAM_ACTIVE
	StreamFinished StreamState = C.TRANSCRIBE_STREAM_FINISHED
	StreamFailed   StreamState = C.TRANSCRIBE_STREAM_FAILED
)

func (s StreamState) String() string {
	switch s {
	case StreamIdle:
		return "idle"
	case StreamActive:
		return "active"
	case StreamFinished:
		return "finished"
	case StreamFailed:
		return "failed"
	}
	return "unknown"
}

// CommitPolicy decides when a stream's committed text is allowed to grow.
type CommitPolicy int

const (
	// CommitAuto uses whatever stable-prefix rule the model's family
	// prefers.
	CommitAuto CommitPolicy = C.TRANSCRIBE_STREAM_COMMIT_AUTO
	// CommitOnFinalize keeps everything tentative until StreamFinalize,
	// which then commits the whole transcript at once.
	CommitOnFinalize CommitPolicy = C.TRANSCRIBE_STREAM_COMMIT_ON_FINALIZE
	// CommitStablePrefix commits a prefix as soon as it stops changing.
	CommitStablePrefix CommitPolicy = C.TRANSCRIBE_STREAM_COMMIT_STABLE_PREFIX
)

// StreamOptions configure a streaming run. The zero value is library
// defaults.
type StreamOptions struct {
	CommitPolicy CommitPolicy
	// AgreementN is how many consecutive hypotheses must agree on a prefix
	// before it is committed, for the policies built on repeated
	// hypotheses. 0 takes the library default, currently 3. Raising it
	// makes a wrong commit less likely and commits later.
	AgreementN int
	// Family carries knobs that only one architecture has, such as
	// MoonshineStreamingOptions. A model that does not accept the kind
	// fails StreamBegin with ErrInvalidArg; Model.AcceptsExtension probes
	// for it.
	Family StreamExtension
}

// StreamBegin starts a streaming run, clearing any previous result on the
// session. Both option sets may be nil for defaults. Streaming needs
// Capabilities.SupportsStreaming; without it this fails rather than falling
// back to the offline path.
//
// Cancelling ctx aborts the stream, which is terminal: the stream moves to
// Failed and StreamLastStatus keeps ErrAborted. ctx has to stay live for the
// whole stream, so unlike Run this does not undo the wiring when it returns;
// StreamFinalize and StreamReset do.
func (s *Session) StreamBegin(ctx context.Context, run *RunOptions, stream *StreamOptions) error {
	rp, free := runParams(run)
	defer free() // the library copies the strings before begin returns

	var sp C.struct_transcribe_stream_params
	C.transcribe_stream_params_init(&sp)
	if stream != nil {
		sp.commit_policy = C.transcribe_stream_commit_policy(stream.CommitPolicy)
		sp.stable_prefix_agreement_n = C.uint32_t(stream.AgreementN)
		if stream.Family != nil {
			ext, freeExt := stream.Family.streamExt()
			sp.family = ext
			defer freeExt() // begin copies what it needs
		}
	}

	if err := check(C.transcribe_stream_begin(s.c, &rp, &sp)); err != nil {
		return err
	}
	s.unwatch()
	s.stopWatch = s.watch(ctx)
	return nil
}

// StreamUpdate is what changed on one feed or finalize call. Its cursors are
// available nowhere else, which is the reason to read it rather than polling
// the session.
type StreamUpdate struct {
	// Changed is true when anything observable moved, and Revision is the
	// counter to diff against the previous call. A bump means "re-read the
	// accessors", not necessarily that the visible text changed.
	Changed  bool
	Revision int
	// Final is true only on the finalize call.
	Final bool
	// CommittedChanged and TentativeChanged say which of the two text
	// views moved.
	CommittedChanged bool
	TentativeChanged bool

	// Received is all audio fed since begin. Committed is the family's
	// report of how far it has got through it, which is a progress hint
	// and not an offset into the committed text. Buffered is what is still
	// held inside the family's streaming state waiting for lookahead, so
	// it is the hint for how much is left to drain.
	Received  time.Duration
	Committed time.Duration
	Buffered  time.Duration
}

func goUpdate(c *C.struct_transcribe_stream_update) StreamUpdate {
	return StreamUpdate{
		Changed:          bool(c.result_changed),
		Revision:         int(c.revision),
		Final:            bool(c.is_final),
		CommittedChanged: bool(c.committed_changed),
		TentativeChanged: bool(c.tentative_changed),
		Received:         ms(c.input_received_ms),
		Committed:        ms(c.audio_committed_ms),
		Buffered:         ms(c.buffered_ms),
	}
}

// StreamFeed pushes PCM into the active stream, in the same mono float32 at
// SampleRate that Run takes. Feeding is the only way to advance a stream:
// the accessors read it without moving it.
//
// A terminal error moves the stream to Failed, where the only ways out are
// StreamReset and another StreamBegin.
func (s *Session) StreamFeed(pcm []float32) (StreamUpdate, error) {
	if len(pcm) == 0 {
		return StreamUpdate{}, ErrInvalidArg
	}
	var u C.struct_transcribe_stream_update
	C.transcribe_stream_update_init(&u)
	err := check(C.transcribe_stream_feed(s.c, (*C.float)(&pcm[0]), C.int(len(pcm)), &u))
	return goUpdate(&u), err
}

// StreamFinalize ends the input, flushes what is buffered, and emits the
// rest of the text. The stream moves to Finished.
func (s *Session) StreamFinalize() (StreamUpdate, error) {
	defer s.unwatch()
	var u C.struct_transcribe_stream_update
	C.transcribe_stream_update_init(&u)
	err := check(C.transcribe_stream_finalize(s.c, &u))
	return goUpdate(&u), err
}

// StreamReset abandons the stream without finalizing and clears the result,
// returning the session to Idle from any state. The family keeps its buffers
// for the next stream; only Close releases those.
func (s *Session) StreamReset() {
	s.unwatch()
	C.transcribe_stream_reset(s.c)
}

// StreamState is the stream's current lifecycle state.
func (s *Session) StreamState() StreamState {
	return StreamState(C.transcribe_stream_get_state(s.c))
}

// StreamRevision is the snapshot counter, which advances whenever anything
// about the streaming result moves. Diff it against the last value you drew
// rather than treating every bump as new text.
func (s *Session) StreamRevision() int { return int(C.transcribe_stream_revision(s.c)) }

// StreamLastStatus is the error that put the stream in Failed, kept so it
// can be read after the fact. It is nil until something fails, and begin and
// reset clear it.
func (s *Session) StreamLastStatus() error {
	return check(C.transcribe_stream_last_status(s.c))
}

// StreamText is the display view of a stream in progress.
//
// Full is the model's current hypothesis and the authoritative one, but any
// of it can be rewritten on the next feed. Committed is append-only, so it
// never flickers, at the cost of being best-effort: a model that re-attends
// over a growing context can revise a byte that was already committed, and
// since Committed is not rolled back, Committed plus Tentative stops
// reconstructing Full when that happens. Render Full for the truth, or
// Committed plus Tentative for a stable prefix.
type StreamText struct {
	Full      string
	Committed string
	Tentative string
}

// StreamText reads the current text snapshot.
func (s *Session) StreamText() (StreamText, error) {
	var t C.struct_transcribe_stream_text
	C.transcribe_stream_text_init(&t)
	if err := check(C.transcribe_stream_get_text(s.c, &t)); err != nil {
		return StreamText{}, err
	}
	return StreamText{
		Full:      C.GoStringN(t.full_text, C.int(t.full_text_bytes)),
		Committed: C.GoStringN(t.committed_text, C.int(t.committed_text_bytes)),
		Tentative: C.GoStringN(t.tentative_text, C.int(t.tentative_text_bytes)),
	}, nil
}

// StreamCommitted is how many raw rows of each kind are behind the committed
// boundary. These are low-level hints: a committed text prefix can drift
// from the raw rows once the model revises something already committed, so
// StreamText is the reliable display contract.
func (s *Session) StreamCommitted() (segments, words, tokens int) {
	return int(C.transcribe_stream_n_committed_segments(s.c)),
		int(C.transcribe_stream_n_committed_words(s.c)),
		int(C.transcribe_stream_n_committed_tokens(s.c))
}

// StreamSnapshot materializes the stream's current hypothesis as a Result,
// for callers that want segments, words or tokens rather than the text views.
// It is a copy taken now, so the next feed does not disturb it.
//
// Most UIs want StreamText instead: it is cheaper, and its committed view is
// the one that does not flicker. RawText is not carried by a streaming
// result and comes back empty.
func (s *Session) StreamSnapshot() Result { return reader{s: s, i: -1}.materialize() }
