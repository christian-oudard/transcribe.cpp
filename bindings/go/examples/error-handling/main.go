// error-handling shows how errors, cancellation and cleanup work.
//
//	go run ./examples/error-handling [model.gguf] [audio.wav]
//
// Every library status is a Status value that satisfies error, so callers
// match with errors.Is rather than parsing strings. The first half needs no
// model and always runs.
package main

import (
	"context"
	"errors"
	"fmt"
	"time"

	transcribe "github.com/handy-computer/transcribe.cpp/bindings/go"
	"github.com/handy-computer/transcribe.cpp/bindings/go/internal/support"
)

func main() {
	// 1. A missing file is a distinct status, not a generic load failure.
	_, err := transcribe.LoadModel("/no/such/model.gguf", nil)
	fmt.Printf("missing file:  errors.Is(err, ErrFileNotFound) = %t (%v)\n",
		errors.Is(err, transcribe.ErrFileNotFound), err)

	model := support.Model(support.Arg(0), support.ModelEnv)
	if model == "" {
		support.Skip("no model, so the run half is skipped (set TRANSCRIBE_SMOKE_MODEL)")
	}
	pcm, err := support.PCM(support.Arg(1))
	if err != nil {
		support.Die(err)
	}

	s, err := transcribe.Open(model, nil, nil)
	if err != nil {
		support.Die(err)
	}
	// Close is idempotent and frees the model too, since Open loaded it.
	defer s.Close()

	// 2. Empty audio is a clean error rather than a panic.
	_, err = s.Run(context.Background(), nil, nil)
	fmt.Printf("empty PCM:     errors.Is(err, ErrInvalidArg) = %t (%v)\n",
		errors.Is(err, transcribe.ErrInvalidArg), err)

	// 3. Asking a model for finer timestamps than it can produce is
	//    reported, not silently downgraded.
	caps, err := s.Model().Capabilities()
	if err != nil {
		support.Die(err)
	}
	if caps.MaxTimestamps < transcribe.StampsToken {
		_, err = s.Run(context.Background(), pcm, &transcribe.RunOptions{Timestamps: transcribe.StampsToken})
		fmt.Printf("token stamps:  errors.Is(err, ErrUnsupportedStamps) = %t (%v)\n",
			errors.Is(err, transcribe.ErrUnsupportedStamps), err)
	}

	// 4. Cancelling the context aborts between decode steps. The result
	//    still holds whatever finished before the abort.
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	res, err := s.Run(ctx, pcm, nil)
	fmt.Printf("cancelled:     errors.Is(err, ErrAborted) = %t, session.Aborted() = %t\n",
		errors.Is(err, transcribe.ErrAborted), s.Aborted())
	if err != nil {
		fmt.Printf("  partial transcript: %q\n", res.Text)
	}

	// 5. A run that hit the model's output budget says so, which is how a
	//    truncated transcript is told from a complete one.
	fmt.Printf("truncated:     %t\n", s.Truncated())
}
