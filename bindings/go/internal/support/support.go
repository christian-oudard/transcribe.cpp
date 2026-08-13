// Package support resolves the fixtures the tests and the canonical examples
// run on, the Go analog of the Rust binding's examples/common.
//
// A model and a clip come from, in order: a command-line argument, the
// TRANSCRIBE_SMOKE_* variable CI exports through fetch-canary, or an in-repo
// default. When nothing resolves, an example prints a skip note and exits 0
// and a test skips, so the whole set runs headless in CI and on forks where
// the canary is absent.
package support

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	"github.com/handy-computer/transcribe.cpp/bindings/go/internal/wav"
)

// The environment variables naming the fixtures, shared with the other
// bindings' test and example runners.
const (
	ModelEnv     = "TRANSCRIBE_SMOKE_MODEL"
	StreamingEnv = "TRANSCRIBE_SMOKE_STREAMING_MODEL"
	AudioEnv     = "TRANSCRIBE_SMOKE_AUDIO"
)

// Arg returns the nth positional argument, or "". Only examples pass these
// on: under `go test` the arguments belong to the test binary, which is why
// the resolvers below take the argument rather than reading os.Args
// themselves.
func Arg(n int) string {
	if args := os.Args[1:]; len(args) > n {
		return args[n]
	}
	return ""
}

// repoRoot is the checkout this package was compiled from. Anchoring on the
// source location rather than the working directory is what lets the tests
// and the examples resolve the same clip while running from different
// directories.
func repoRoot() string {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		return "."
	}
	// .../bindings/go/internal/support/support.go -> the checkout root.
	return filepath.Clean(filepath.Join(filepath.Dir(file), "..", "..", "..", ".."))
}

// Model resolves the model path from arg, else env, else "". Streaming
// callers pass StreamingEnv, since not every family streams.
func Model(arg, env string) string {
	if arg != "" {
		return arg
	}
	return os.Getenv(env)
}

// Audio resolves the clip to transcribe, defaulting to the sample in the
// repo, which is what every binding's tests and examples use.
func Audio(arg string) string {
	if arg != "" {
		return arg
	}
	if p := os.Getenv(AudioEnv); p != "" {
		return p
	}
	return filepath.Join(repoRoot(), "samples", "jfk.wav")
}

// PCM reads the resolved clip.
func PCM(arg string) ([]float32, error) { return wav.Read(Audio(arg)) }

// Skip reports that a fixture is missing and exits 0, so a CI run without the
// canary models is not a failure.
func Skip(msg string) {
	fmt.Println("skip:", msg)
	os.Exit(0)
}

// Die reports a real failure.
func Die(err error) {
	fmt.Fprintln(os.Stderr, "error:", err)
	os.Exit(1)
}
