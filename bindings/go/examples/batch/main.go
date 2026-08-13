// batch transcribes several utterances in one call and shows the
// per-utterance error contract: the batch can succeed with one slot failing.
//
//	go run ./examples/batch [model.gguf] [audio.wav]
package main

import (
	"context"
	"fmt"
	"time"

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

	s, err := transcribe.Open(model, nil, nil)
	if err != nil {
		support.Die(err)
	}
	defer s.Close()

	// The clip, its first half, and an empty utterance that fails on its
	// own without taking the batch down with it.
	utterances := [][]float32{pcm, pcm[:len(pcm)/2], nil}

	start := time.Now()
	res, err := s.RunBatch(context.Background(), utterances, nil)
	if err != nil {
		support.Die(err)
	}
	batched := time.Since(start)

	for i, r := range res {
		if r.Err != nil {
			fmt.Printf("utterance %d failed: %v\n", i, r.Err)
			continue
		}
		fmt.Printf("utterance %d: %s\n", i, r.Text)
	}

	// The same work one at a time, for comparison. Families with a batched
	// compute path win here; the rest run them in turn either way.
	start = time.Now()
	for _, u := range utterances[:2] {
		if _, err := s.Run(context.Background(), u, nil); err != nil {
			support.Die(err)
		}
	}
	fmt.Printf("\nbatched %v, one at a time %v\n", batched, time.Since(start))
}
