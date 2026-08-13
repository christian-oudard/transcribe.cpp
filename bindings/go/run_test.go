package transcribe

// End-to-end tests. They need a GGUF model, which is too big to keep in the
// tree and must not be downloaded from a test, so they skip unless
// TRANSCRIBE_SMOKE_MODEL points at one:
//
//	TRANSCRIBE_SMOKE_MODEL=~/models/moonshine-tiny-Q8_0.gguf go test ./...

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/handy-computer/transcribe.cpp/bindings/go/internal/support"
)

// modelPath is the model named by env, or a skip when there is none. The
// tests and the examples resolve fixtures the same way, so a checkout with
// no canary models skips rather than failing.
func modelPath(t *testing.T, env string) string {
	t.Helper()
	path := support.Model("", env)
	if path == "" {
		t.Skipf("set %s to a .gguf to run this", env)
	}
	return path
}

// testSession opens a session on the offline model, or skips.
func testSession(t *testing.T, load *LoadOptions) *Session {
	t.Helper()
	path := modelPath(t, support.ModelEnv)
	s, err := Open(path, load, nil)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	t.Cleanup(s.Close)
	return s
}

// jfk is the sample clip every binding's tests transcribe. Decoding it once
// keeps it out of the per-test cost; nothing mutates the samples.
var jfkOnce = sync.OnceValues(func() ([]float32, error) { return support.PCM("") })

func jfk(t *testing.T) []float32 {
	t.Helper()
	pcm, err := jfkOnce()
	if err != nil {
		t.Fatal(err)
	}
	return pcm
}

func TestRun(t *testing.T) {
	s := testSession(t, nil)
	res, err := s.Run(t.Context(), jfk(t), nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Log(res.Text)
	if !strings.Contains(strings.ToLower(res.Text), "fellow americans") {
		t.Errorf("transcript does not look like the JFK sample: %q", res.Text)
	}
	if s.Aborted() {
		t.Error("a run that completed reports itself aborted")
	}

	t.Logf("load %v, encode %v, decode %v", res.Timings.Load, res.Timings.Encode, res.Timings.Decode)
	if res.Timings.Load <= 0 {
		t.Error("no load time recorded")
	}
	// Raw text is the decode before family post-processing, so it carries
	// the same words modulo whitespace and markers.
	if res.RawText == "" {
		t.Error("no raw text")
	}
}

func TestRunSegments(t *testing.T) {
	s := testSession(t, nil)
	res, err := s.Run(t.Context(), jfk(t), nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Segments) == 0 {
		t.Fatal("no segments")
	}
	var parts []string
	for _, seg := range res.Segments {
		t.Logf("[%v-%v] %s", seg.Start, seg.End, seg.Text)
		parts = append(parts, seg.Text)
	}
	// Segments partition the transcript, so they reconstruct it. Families
	// trim each segment's own whitespace, so compare on words rather than
	// on the exact bytes.
	words := func(s string) string { return strings.Join(strings.Fields(s), " ") }
	if got := words(strings.Join(parts, " ")); got != words(res.Text) {
		t.Errorf("segments do not reconstruct the text:\n %q\n %q", got, res.Text)
	}
	if res.Timestamps == StampsAuto {
		t.Error("the run reported Auto rather than the granularity it resolved to")
	}
}

// TestRunFinestStamps asks for the finest granularity the model advertises
// and checks the row cross-references hold, since the index fields on a
// segment are only useful if they really index the sibling slices.
func TestRunFinestStamps(t *testing.T) {
	s := testSession(t, nil)
	caps, err := s.Model().Capabilities()
	if err != nil {
		t.Fatal(err)
	}
	if caps.MaxTimestamps == StampsNone {
		t.Skipf("%s produces no timestamps", s.Model().Variant())
	}

	res, err := s.Run(t.Context(), jfk(t), &RunOptions{Timestamps: caps.MaxTimestamps})
	if err != nil {
		t.Fatal(err)
	}
	if res.Timestamps != caps.MaxTimestamps {
		t.Errorf("asked for %v, got %v", caps.MaxTimestamps, res.Timestamps)
	}
	for i, seg := range res.Segments {
		if seg.End < seg.Start {
			t.Errorf("segment %d ends before it starts: %v to %v", i, seg.Start, seg.End)
		}
		if seg.NumWords > 0 && seg.FirstWord+seg.NumWords > len(res.Words) {
			t.Errorf("segment %d spans words %d..%d of %d",
				i, seg.FirstWord, seg.FirstWord+seg.NumWords, len(res.Words))
		}
		if seg.NumTokens > 0 && seg.FirstToken+seg.NumTokens > len(res.Tokens) {
			t.Errorf("segment %d spans tokens %d..%d of %d",
				i, seg.FirstToken, seg.FirstToken+seg.NumTokens, len(res.Tokens))
		}
	}
	for i, w := range res.Words {
		if w.Segment < 0 || w.Segment >= len(res.Segments) {
			t.Errorf("word %d belongs to segment %d of %d", i, w.Segment, len(res.Segments))
		}
	}
	t.Logf("%v: %d segments, %d words, %d tokens",
		res.Timestamps, len(res.Segments), len(res.Words), len(res.Tokens))
}

// TestResultOutlivesRun covers the contract every binding in this repo keeps:
// a result owns its contents, so the next run does not disturb it. The
// library's own accessors hand back pointers into session storage that the
// next run overwrites, which is exactly what materializing avoids.
func TestResultOutlivesRun(t *testing.T) {
	s := testSession(t, nil)
	pcm := jfk(t)
	first, err := s.Run(t.Context(), pcm, nil)
	if err != nil {
		t.Fatal(err)
	}
	text, nseg := first.Text, len(first.Segments)
	var segText string
	if nseg > 0 {
		segText = first.Segments[0].Text
	}

	// Run again on a prefix, so the second result is definitely different.
	if _, err := s.Run(t.Context(), pcm[:len(pcm)/3], nil); err != nil {
		t.Fatal(err)
	}
	if first.Text != text {
		t.Errorf("the first result's text changed under it:\n %q\n %q", text, first.Text)
	}
	if len(first.Segments) != nseg {
		t.Errorf("the first result's segments changed under it: %d then %d", nseg, len(first.Segments))
	}
	if nseg > 0 && first.Segments[0].Text != segText {
		t.Errorf("a segment's text changed under it:\n %q\n %q", segText, first.Segments[0].Text)
	}

	// A result also outlives the session it came from, since nothing in it
	// points at the session any more.
	s.Close()
	if first.Text != text {
		t.Error("the result did not survive closing the session")
	}
}

func TestRunEmptyPCM(t *testing.T) {
	s := testSession(t, nil)
	if _, err := s.Run(t.Context(), nil, nil); !errors.Is(err, ErrInvalidArg) {
		t.Fatalf("want ErrInvalidArg for empty audio, got %v", err)
	}
}

func TestRunCancelled(t *testing.T) {
	s := testSession(t, nil)
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	_, err := s.Run(ctx, jfk(t), nil)
	if !errors.Is(err, ErrAborted) {
		t.Fatalf("want ErrAborted from a cancelled context, got %v", err)
	}
	if !s.Aborted() {
		t.Error("session does not report the abort")
	}
}

func TestRunBatch(t *testing.T) {
	s := testSession(t, nil)
	pcm := jfk(t)
	// The same clip twice: whatever the family does with batching, the two
	// results have to agree.
	res, err := s.RunBatch(t.Context(), [][]float32{pcm, pcm}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 2 {
		t.Fatalf("want 2 results, got %d", len(res))
	}
	for i, r := range res {
		if r.Err != nil {
			t.Fatalf("utterance %d: %v", i, r.Err)
		}
	}
	if res[0].Text != res[1].Text {
		t.Errorf("identical audio transcribed differently:\n %q\n %q", res[0].Text, res[1].Text)
	}
	if res[0].Text == "" {
		t.Error("empty transcript")
	}
}

func TestBatchPerUtteranceFailure(t *testing.T) {
	s := testSession(t, nil)
	// A malformed utterance inside a valid batch fails only itself, so the
	// call succeeds and the bad slot carries the error.
	res, err := s.RunBatch(t.Context(), [][]float32{jfk(t), nil}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res[0].Err != nil {
		t.Errorf("good utterance failed: %v", res[0].Err)
	}
	if res[0].Text == "" {
		t.Error("the good utterance produced no text")
	}
	if !errors.Is(res[1].Err, ErrInvalidArg) {
		t.Errorf("want ErrInvalidArg for the empty utterance, got %v", res[1].Err)
	}
	// A slot that failed outright carries no transcript. Reading one back
	// would mean reading whatever was in that storage before.
	if res[1].Text != "" {
		t.Errorf("the failed utterance carries text: %q", res[1].Text)
	}
}

// TestRunFailureKeepsNoStaleResult checks that a failed run does not hand
// back the previous run's transcript, which is what reading the session's
// result storage unconditionally would do.
func TestRunFailureKeepsNoStaleResult(t *testing.T) {
	s := testSession(t, nil)
	first, err := s.Run(t.Context(), jfk(t), nil)
	if err != nil {
		t.Fatal(err)
	}
	if first.Text == "" {
		t.Fatal("no transcript to go stale")
	}
	// Empty audio fails before the model runs, leaving the previous result
	// in place inside the session.
	res, err := s.Run(t.Context(), nil, nil)
	if !errors.Is(err, ErrInvalidArg) {
		t.Fatalf("want ErrInvalidArg, got %v", err)
	}
	if res.Text != "" {
		t.Errorf("a failed run returned the previous transcript: %q", res.Text)
	}
}

func TestModelIntrospection(t *testing.T) {
	s := testSession(t, nil)
	m := s.Model()
	if m.Arch() == "" {
		t.Error("no architecture")
	}
	t.Logf("%s / %s on %s", m.Arch(), m.Variant(), m.Backend())

	caps, err := m.Capabilities()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%+v", caps)
	if caps.NativeSampleRate == 0 {
		t.Error("no native sample rate")
	}
	if caps.MaxTimestamps == StampsAuto {
		t.Error("Auto is not a granularity a model can max out at")
	}

	dev, err := m.Device()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("running on %s (%s)", dev.Name, dev.Type)

	if _, err := s.Limits(); err != nil {
		t.Fatal(err)
	}

	// The session owns this model, so closing it must not free anything;
	// the session's own Close, from t.Cleanup, would then double free.
	m.Close()
	if m.Arch() == "" {
		t.Error("closing a borrowed model freed it")
	}
}

func TestTokenize(t *testing.T) {
	s := testSession(t, nil)
	m := s.Model()
	tokens, err := m.Tokenize("ask not what your country can do for you")
	if errors.Is(err, ErrNoTokenizer) {
		t.Skipf("%s has no encode path", m.Variant())
	}
	if err != nil {
		t.Fatal(err)
	}
	if len(tokens) == 0 {
		t.Fatal("no tokens for a non-empty string")
	}
	t.Logf("%d tokens: %v", len(tokens), tokens)

	// Longer text has to produce at least as many tokens, whatever the
	// vocabulary; anything else means the length handshake went wrong.
	more, err := m.Tokenize("ask not what your country can do for you, ask what you can do for your country")
	if err != nil {
		t.Fatal(err)
	}
	if len(more) <= len(tokens) {
		t.Errorf("longer text made %d tokens against %d", len(more), len(tokens))
	}
}

// TestSharedModel covers the two-step path: one model, several sessions.
// They are used one at a time, which is the only supported way to share a
// model: the library does not allow concurrent compute across its sessions.
func TestSharedModel(t *testing.T) {
	m, err := LoadModel(modelPath(t, support.ModelEnv), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	pcm := jfk(t)
	var texts []string
	for range 2 {
		s, err := m.NewSession(nil)
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		res, err := s.Run(t.Context(), pcm, nil)
		if err != nil {
			t.Fatal(err)
		}
		texts = append(texts, res.Text)
	}
	if texts[0] != texts[1] {
		t.Errorf("sessions on one model disagreed:\n %q\n %q", texts[0], texts[1])
	}
}
