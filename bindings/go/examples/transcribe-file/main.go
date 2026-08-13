// transcribe-file transcribes one clip and prints the transcript and its
// segments.
//
//	go run ./examples/transcribe-file [model.gguf] [audio.wav]
package main

import (
	"context"
	"fmt"

	transcribe "github.com/handy-computer/transcribe.cpp/bindings/go"
	"github.com/handy-computer/transcribe.cpp/bindings/go/internal/support"
)

func main() {
	model := support.Model(support.Arg(0), support.ModelEnv)
	if model == "" {
		support.Skip("no model (pass one, or set TRANSCRIBE_SMOKE_MODEL)")
	}
	pcm, err := support.PCM(support.Arg(1))
	if err != nil {
		support.Die(err)
	}

	// Open loads the model and opens one session against it. Close frees
	// both, and the result outlives them.
	s, err := transcribe.Open(model, nil, nil)
	if err != nil {
		support.Die(err)
	}
	defer s.Close()

	res, err := s.Run(context.Background(), pcm, nil)
	if err != nil {
		support.Die(err)
	}

	m := s.Model()
	fmt.Printf("model:    %s / %s on %s\n", m.Arch(), m.Variant(), m.Backend())
	fmt.Printf("language: %s\n", res.Language)
	fmt.Printf("stamps:   %s\n\n", res.Timestamps)
	fmt.Println(res.Text)

	if len(res.Segments) > 0 {
		fmt.Println("\nsegments:")
		for _, seg := range res.Segments {
			fmt.Printf("  [%6v - %6v] %s\n", seg.Start, seg.End, seg.Text)
		}
	}
	fmt.Printf("\nencode %v, decode %v\n", res.Timings.Encode, res.Timings.Decode)
}
