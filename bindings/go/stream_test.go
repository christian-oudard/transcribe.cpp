package transcribe

// Streaming tests need a model that streams, which not every family does, so
// they take their own environment variable:
//
//	TRANSCRIBE_SMOKE_STREAMING_MODEL=~/models/moonshine-streaming-tiny-Q8_0.gguf go test ./...

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/handy-computer/transcribe.cpp/bindings/go/internal/support"
)

func testStreamSession(t *testing.T) *Session {
	t.Helper()
	path := modelPath(t, support.StreamingEnv)
	s, err := Open(path, nil, nil)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	t.Cleanup(s.Close)

	caps, err := s.Model().Capabilities()
	if err != nil {
		t.Fatal(err)
	}
	if !caps.SupportsStreaming {
		t.Fatalf("%s does not stream", path)
	}
	return s
}

// feedAll pushes the clip through the stream in chunk-sized slices, the way
// a live capture would arrive.
func feedAll(t *testing.T, s *Session, pcm []float32, chunk int) []StreamUpdate {
	t.Helper()
	var updates []StreamUpdate
	for off := 0; off < len(pcm); off += chunk {
		end := min(off+chunk, len(pcm))
		u, err := s.StreamFeed(pcm[off:end])
		if err != nil {
			t.Fatalf("feed at %d samples: %v (stream %v)", off, err, s.StreamState())
		}
		updates = append(updates, u)
	}
	return updates
}

func TestStream(t *testing.T) {
	s := testStreamSession(t)
	if got := s.StreamState(); got != StreamIdle {
		t.Fatalf("a fresh session is %v, want idle", got)
	}
	if err := s.StreamBegin(t.Context(), nil, nil); err != nil {
		t.Fatal(err)
	}
	if got := s.StreamState(); got != StreamActive {
		t.Fatalf("after begin the stream is %v, want active", got)
	}

	pcm := jfk(t)
	// Half a second at a time, which is the order a dictation UI feeds at.
	updates := feedAll(t, s, pcm, SampleRate/2)

	final, err := s.StreamFinalize()
	if err != nil {
		t.Fatal(err)
	}
	if !final.Final {
		t.Error("the finalize update is not marked final")
	}
	if got := s.StreamState(); got != StreamFinished {
		t.Fatalf("after finalize the stream is %v, want finished", got)
	}

	text, err := s.StreamText()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("full:      %s", text.Full)
	t.Logf("committed: %s", text.Committed)
	if !strings.Contains(strings.ToLower(text.Full), "fellow americans") {
		t.Errorf("transcript does not look like the JFK sample: %q", text.Full)
	}
	// Finalize empties the tentative view by definition.
	if text.Tentative != "" {
		t.Errorf("tentative text survived finalize: %q", text.Tentative)
	}
	// Received counts every sample fed, so it should match the clip.
	wantMS := int64(len(pcm)) * 1000 / SampleRate
	if got := final.Received.Milliseconds(); got < wantMS-100 || got > wantMS+100 {
		t.Errorf("stream received %d ms of a %d ms clip", got, wantMS)
	}
	if s.StreamLastStatus() != nil {
		t.Errorf("a stream that finished cleanly reports %v", s.StreamLastStatus())
	}

	// The row view of the same hypothesis, for callers that want segments
	// rather than the text views.
	snap := s.StreamSnapshot()
	if snap.Text != text.Full {
		t.Errorf("snapshot text differs from the stream's:\n %q\n %q", snap.Text, text.Full)
	}
	if len(snap.Segments) == 0 {
		t.Error("no segments in the snapshot")
	}

	// The revision counter has to have moved, or nothing was observable.
	if s.StreamRevision() == 0 {
		t.Error("the revision counter never advanced")
	}
	var advanced bool
	for _, u := range updates {
		if u.Changed {
			advanced = true
		}
	}
	if !advanced {
		t.Error("no feed reported a change")
	}
}

// TestStreamCommittedGrows covers the append-only contract that makes the
// committed view usable for display: it may lag, but it never shrinks or
// rewrites what it already showed.
func TestStreamCommittedGrows(t *testing.T) {
	s := testStreamSession(t)
	if err := s.StreamBegin(t.Context(), nil, &StreamOptions{CommitPolicy: CommitStablePrefix}); err != nil {
		t.Fatal(err)
	}
	pcm := jfk(t)
	var prev string
	for off := 0; off < len(pcm); off += SampleRate / 2 {
		end := min(off+SampleRate/2, len(pcm))
		if _, err := s.StreamFeed(pcm[off:end]); err != nil {
			t.Fatal(err)
		}
		text, err := s.StreamText()
		if err != nil {
			t.Fatal(err)
		}
		if !strings.HasPrefix(text.Committed, prev) {
			t.Fatalf("committed text was rewritten:\n was %q\n now %q", prev, text.Committed)
		}
		prev = text.Committed
	}
	if _, err := s.StreamFinalize(); err != nil {
		t.Fatal(err)
	}
	text, err := s.StreamText()
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(text.Committed, prev) {
		t.Errorf("finalize rewrote committed text:\n was %q\n now %q", prev, text.Committed)
	}
}

// TestStreamOnFinalize checks the other policy: nothing is committed until
// the end, so a UI that cannot tolerate a wrong commit sees text once.
func TestStreamOnFinalize(t *testing.T) {
	s := testStreamSession(t)
	if err := s.StreamBegin(t.Context(), nil, &StreamOptions{CommitPolicy: CommitOnFinalize}); err != nil {
		t.Fatal(err)
	}
	pcm := jfk(t)
	feedAll(t, s, pcm, SampleRate/2)

	text, err := s.StreamText()
	if err != nil {
		t.Fatal(err)
	}
	if text.Committed != "" {
		t.Errorf("committed text appeared before finalize: %q", text.Committed)
	}
	if _, err := s.StreamFinalize(); err != nil {
		t.Fatal(err)
	}
	text, err = s.StreamText()
	if err != nil {
		t.Fatal(err)
	}
	if text.Committed == "" {
		t.Error("finalize committed nothing")
	}
	if text.Committed != text.Full {
		t.Errorf("finalize committed something other than the transcript:\n %q\n %q", text.Committed, text.Full)
	}
}

func TestStreamReset(t *testing.T) {
	s := testStreamSession(t)
	if err := s.StreamBegin(t.Context(), nil, nil); err != nil {
		t.Fatal(err)
	}
	feedAll(t, s, jfk(t), SampleRate)
	s.StreamReset()

	if got := s.StreamState(); got != StreamIdle {
		t.Fatalf("after reset the stream is %v, want idle", got)
	}
	if s.StreamRevision() != 0 {
		t.Error("reset left the revision counter behind")
	}
	text, err := s.StreamText()
	if err != nil {
		t.Fatal(err)
	}
	if text.Full != "" {
		t.Errorf("reset left a transcript behind: %q", text.Full)
	}
	// A reset session takes a new stream.
	if err := s.StreamBegin(t.Context(), nil, nil); err != nil {
		t.Fatal(err)
	}
}

func TestStreamFeedWhenNotActive(t *testing.T) {
	s := testStreamSession(t)
	if _, err := s.StreamFeed(jfk(t)); !errors.Is(err, ErrInvalidArg) {
		t.Fatalf("want ErrInvalidArg feeding an idle session, got %v", err)
	}
	if _, err := s.StreamFinalize(); !errors.Is(err, ErrInvalidArg) {
		t.Fatalf("want ErrInvalidArg finalizing an idle session, got %v", err)
	}
}

func TestStreamCancel(t *testing.T) {
	s := testStreamSession(t)
	ctx, cancel := context.WithCancel(t.Context())
	if err := s.StreamBegin(ctx, nil, nil); err != nil {
		t.Fatal(err)
	}
	pcm := jfk(t)
	if _, err := s.StreamFeed(pcm[:SampleRate]); err != nil {
		t.Fatal(err)
	}
	cancel()

	// Abort is terminal for a stream, so the failure can land on this feed
	// or on the finalize that follows it.
	_, feedErr := s.StreamFeed(pcm[SampleRate : 2*SampleRate])
	_, finalErr := s.StreamFinalize()
	if !errors.Is(feedErr, ErrAborted) && !errors.Is(finalErr, ErrAborted) {
		t.Fatalf("cancelling did not abort the stream: feed %v, finalize %v", feedErr, finalErr)
	}
	if got := s.StreamState(); got != StreamFailed {
		t.Errorf("an aborted stream is %v, want failed", got)
	}
	if !errors.Is(s.StreamLastStatus(), ErrAborted) {
		t.Errorf("last status is %v, want ErrAborted", s.StreamLastStatus())
	}
	if !s.Aborted() {
		t.Error("session does not report the abort")
	}
}

// TestStreamThenRun checks that the two paths do not trip over each other:
// a run during an active stream is refused, and works again after a reset.
func TestStreamThenRun(t *testing.T) {
	s := testStreamSession(t)
	if err := s.StreamBegin(t.Context(), nil, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Run(t.Context(), jfk(t), nil); !errors.Is(err, ErrInvalidArg) {
		t.Fatalf("want ErrInvalidArg running during a stream, got %v", err)
	}
	s.StreamReset()
	if _, err := s.Run(t.Context(), jfk(t), nil); err != nil {
		t.Fatalf("run after reset: %v", err)
	}
}
