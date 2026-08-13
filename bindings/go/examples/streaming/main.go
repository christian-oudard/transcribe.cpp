// streaming feeds a clip in real-time-sized chunks and redraws the transcript
// as it arrives, the way a dictation UI would.
//
//	go run ./examples/streaming [streaming-model.gguf] [audio.wav]
package main

import (
	"context"
	"fmt"

	transcribe "github.com/handy-computer/transcribe.cpp/bindings/go"
	"github.com/handy-computer/transcribe.cpp/bindings/go/internal/support"
)

func main() {
	model := support.Model(support.Arg(0), support.StreamingEnv)
	if model == "" {
		support.Skip("no streaming model (pass one, or set TRANSCRIBE_SMOKE_STREAMING_MODEL)")
	}
	pcm, err := support.PCM(support.Arg(1))
	if err != nil {
		support.Die(err)
	}

	s, err := transcribe.Open(model, nil, nil)
	if err != nil {
		support.Die(err)
	}
	defer s.Close()

	// Not every family streams, so ask before beginning rather than
	// interpreting the error.
	m := s.Model()
	caps, err := m.Capabilities()
	if err != nil {
		support.Die(err)
	}
	if !caps.SupportsStreaming {
		support.Skip(fmt.Sprintf("%s does not stream", m.Variant()))
	}

	if err := s.StreamBegin(context.Background(), nil, nil); err != nil {
		support.Die(err)
	}

	// Half a second per feed, the granularity a capture callback delivers.
	const chunk = transcribe.SampleRate / 2
	for off := 0; off < len(pcm); off += chunk {
		end := min(off+chunk, len(pcm))
		u, err := s.StreamFeed(pcm[off:end])
		if err != nil {
			support.Die(err)
		}
		if !u.Changed {
			continue
		}
		text, err := s.StreamText()
		if err != nil {
			support.Die(err)
		}
		// Committed never flickers; tentative is the volatile tail. A UI
		// renders the first as settled text and the second as a preview.
		fmt.Printf("[%5v] %s\033[2m%s\033[0m\n", u.Received, text.Committed, text.Tentative)
	}

	final, err := s.StreamFinalize()
	if err != nil {
		support.Die(err)
	}
	text, err := s.StreamText()
	if err != nil {
		support.Die(err)
	}
	fmt.Printf("\nfinal (revision %d, %v of audio):\n%s\n", final.Revision, final.Received, text.Full)
}
