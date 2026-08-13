package transcribe

// Family extension tests. Each kind is accepted by one family in one slot,
// so these need the matching model and skip without it.

import (
	"errors"
	"strings"
	"testing"

	"github.com/handy-computer/transcribe.cpp/bindings/go/internal/support"
)

func TestOpt(t *testing.T) {
	if got := Opt[float32](0.25); got == nil || *got != 0.25 {
		t.Fatalf("Opt did not carry the value: %v", got)
	}
	// The point of the wrapper is that a set zero is distinguishable from
	// an unset field.
	if got := Opt(0); got == nil || *got != 0 {
		t.Fatal("Opt lost a zero")
	}
}

// TestExtensionKindsAreDistinct guards the registry: two kinds sharing a
// value would send one family's struct to another's handler.
func TestExtensionKindsAreDistinct(t *testing.T) {
	kinds := map[ExtKind]string{}
	for name, k := range map[string]ExtKind{
		"whisper run":             KindWhisperRun,
		"sortformer stream":       KindSortformerStream,
		"parakeet stream":         KindParakeetStream,
		"parakeet buffered":       KindParakeetBufferedStream,
		"moonshine streaming":     KindMoonshineStreaming,
		"voxtral realtime stream": KindVoxtralRealtimeStream,
	} {
		if other, dup := kinds[k]; dup {
			t.Errorf("%s and %s share kind %#x", name, other, uint32(k))
		}
		kinds[k] = name
	}
}

// TestExtensionSlots pins each options type to its slot. The compile-time
// half is stronger than the test: a type that does not implement the slot's
// interface cannot be assigned to that Family field at all.
func TestExtensionSlots(t *testing.T) {
	var run []RunExtension = []RunExtension{
		&WhisperRunOptions{}, &SortformerStreamOptions{},
	}
	var stream []StreamExtension = []StreamExtension{
		&ParakeetStreamOptions{}, &ParakeetBufferedStreamOptions{},
		&MoonshineStreamingOptions{}, &VoxtralRealtimeStreamOptions{},
	}
	for _, e := range run {
		if e.Kind() == 0 {
			t.Errorf("%T reports no kind", e)
		}
	}
	for _, e := range stream {
		if e.Kind() == 0 {
			t.Errorf("%T reports no kind", e)
		}
	}
}

func TestAcceptsExtension(t *testing.T) {
	s := testSession(t, nil)
	m := s.Model()
	arch := m.Arch()

	accepted := m.AcceptsExtension(SlotRun, KindWhisperRun)
	t.Logf("%s accepts the whisper run extension: %t", arch, accepted)
	if arch == "whisper" && !accepted {
		t.Error("a whisper model rejects the whisper run extension")
	}
	if arch != "whisper" && accepted {
		t.Errorf("%s accepts the whisper run extension", arch)
	}
	// A kind is legal in exactly one slot, so the same kind in the other
	// slot is never accepted.
	if m.AcceptsExtension(SlotStream, KindWhisperRun) {
		t.Error("the whisper run extension was accepted on the stream slot")
	}
}

func TestWhisperRunExtension(t *testing.T) {
	s := testSession(t, nil)
	if !s.Model().AcceptsExtension(SlotRun, KindWhisperRun) {
		t.Skipf("%s takes no whisper run extension", s.Model().Variant())
	}

	// A greedy decode with the fallback ladder pinned to one tier, which
	// is the reproducible configuration.
	res, err := s.Run(t.Context(), jfk(t), &RunOptions{
		Family: &WhisperRunOptions{
			Temperature:         Opt[float32](0),
			TemperatureInc:      Opt[float32](0),
			MaxInitialTimestamp: Opt[float32](1),
			Seed:                Opt[uint32](1234),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(strings.ToLower(res.Text), "fellow americans") {
		t.Errorf("transcript does not look like the JFK sample: %q", res.Text)
	}
}

// TestWhisperInitialPrompt checks the prompt reaches the decoder. The
// library rejects a special tag inside one, which is the observable proof
// that the text was tokenized as a prompt rather than ignored.
func TestWhisperInitialPrompt(t *testing.T) {
	s := testSession(t, nil)
	if !s.Model().AcceptsExtension(SlotRun, KindWhisperRun) {
		t.Skipf("%s takes no whisper run extension", s.Model().Variant())
	}

	res, err := s.Run(t.Context(), jfk(t), &RunOptions{
		Family: &WhisperRunOptions{
			InitialPrompt:         "A presidential inaugural address.",
			PromptCondition:       Opt(PromptAllSegments),
			ConditionOnPrevTokens: Opt(true),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Text == "" {
		t.Error("a prompted run produced nothing")
	}

	_, err = s.Run(t.Context(), jfk(t), &RunOptions{
		Family: &WhisperRunOptions{InitialPrompt: "<|startoftranscript|> nope"},
	})
	if !errors.Is(err, ErrInvalidArg) {
		t.Errorf("want ErrInvalidArg for a special tag in the prompt, got %v", err)
	}
}

// TestWhisperPromptConditionNeedsPrevTokens pins the cross-field constraint
// the library enforces for HF parity, since nothing in the type system says
// the two fields are related.
func TestWhisperPromptConditionNeedsPrevTokens(t *testing.T) {
	s := testSession(t, nil)
	if !s.Model().AcceptsExtension(SlotRun, KindWhisperRun) {
		t.Skipf("%s takes no whisper run extension", s.Model().Variant())
	}
	_, err := s.Run(t.Context(), jfk(t), &RunOptions{
		Family: &WhisperRunOptions{PromptCondition: Opt(PromptAllSegments)},
	})
	if !errors.Is(err, ErrInvalidArg) {
		t.Errorf("want ErrInvalidArg for all-segments without prev tokens, got %v", err)
	}
}

// TestWhisperPromptTokens covers the already-tokenized path, which takes
// precedence over the text one.
func TestWhisperPromptTokens(t *testing.T) {
	s := testSession(t, nil)
	m := s.Model()
	if !m.AcceptsExtension(SlotRun, KindWhisperRun) {
		t.Skipf("%s takes no whisper run extension", m.Variant())
	}
	tokens, err := m.Tokenize("a presidential inaugural address")
	if errors.Is(err, ErrNoTokenizer) {
		t.Skip("model cannot tokenize, so there are no tokens to pass")
	}
	if err != nil {
		t.Fatal(err)
	}

	res, err := s.Run(t.Context(), jfk(t), &RunOptions{
		Family: &WhisperRunOptions{PromptTokens: tokens},
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Text == "" {
		t.Error("a token-prompted run produced nothing")
	}
}

// TestExtensionWrongFamily checks the library rejects an extension the model
// does not accept rather than ignoring it.
func TestExtensionWrongFamily(t *testing.T) {
	s := testSession(t, nil)
	if s.Model().AcceptsExtension(SlotRun, KindSortformerStream) {
		t.Skip("this model does take the sortformer extension")
	}
	_, err := s.Run(t.Context(), jfk(t), &RunOptions{
		Family: &SortformerStreamOptions{Preset: Opt(SortformerLowLatency)},
	})
	if !errors.Is(err, ErrInvalidArg) {
		t.Errorf("want ErrInvalidArg for an extension this model rejects, got %v", err)
	}
}

func TestMoonshineStreamExtension(t *testing.T) {
	s := testStreamSession(t)
	if !s.Model().AcceptsExtension(SlotStream, KindMoonshineStreaming) {
		t.Skipf("%s takes no moonshine streaming extension", s.Model().Variant())
	}

	// A floor on the re-decode rate: the transcript must still come out.
	err := s.StreamBegin(t.Context(), nil, &StreamOptions{
		Family: &MoonshineStreamingOptions{MinDecodeIntervalMS: Opt(250)},
	})
	if err != nil {
		t.Fatal(err)
	}
	feedAll(t, s, jfk(t), SampleRate/2)
	if _, err := s.StreamFinalize(); err != nil {
		t.Fatal(err)
	}
	text, err := s.StreamText()
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(strings.ToLower(text.Full), "fellow americans") {
		t.Errorf("transcript does not look like the JFK sample: %q", text.Full)
	}
}

// TestStreamExtensionWrongFamily is the streaming half of the rejection
// contract.
func TestStreamExtensionWrongFamily(t *testing.T) {
	s := testStreamSession(t)
	if s.Model().AcceptsExtension(SlotStream, KindVoxtralRealtimeStream) {
		t.Skip("this model does take the voxtral extension")
	}
	err := s.StreamBegin(t.Context(), nil, &StreamOptions{
		Family: &VoxtralRealtimeStreamOptions{NumDelayTokens: Opt(4)},
	})
	if !errors.Is(err, ErrInvalidArg) {
		t.Errorf("want ErrInvalidArg for an extension this model rejects, got %v", err)
	}
}

// TestParakeetExtensions runs only where a parakeet canary is present, which
// is the only place the two parakeet schemas can be told apart.
func TestParakeetExtensions(t *testing.T) {
	path := support.Model("", "TRANSCRIBE_SMOKE_PARAKEET_STREAM_MODEL")
	if path == "" {
		t.Skip("set TRANSCRIBE_SMOKE_PARAKEET_STREAM_MODEL to run this")
	}
	s, err := Open(path, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	m := s.Model()
	cacheAware := m.AcceptsExtension(SlotStream, KindParakeetStream)
	buffered := m.AcceptsExtension(SlotStream, KindParakeetBufferedStream)
	t.Logf("%s: cache-aware %t, buffered %t", m.Variant(), cacheAware, buffered)
	// The two schemas are for different variants, so a model takes one.
	if cacheAware == buffered {
		t.Errorf("a parakeet variant accepts both schemas or neither")
	}

	var ext StreamExtension = &ParakeetStreamOptions{AttContextRight: Opt(2)}
	if buffered {
		ext = &ParakeetBufferedStreamOptions{ChunkMS: Opt(1600)}
	}
	if err := s.StreamBegin(t.Context(), nil, &StreamOptions{Family: ext}); err != nil {
		t.Fatal(err)
	}
	feedAll(t, s, jfk(t), SampleRate/2)
	if _, err := s.StreamFinalize(); err != nil {
		t.Fatal(err)
	}
	text, err := s.StreamText()
	if err != nil {
		t.Fatal(err)
	}
	if text.Full == "" {
		t.Error("empty transcript")
	}
}
