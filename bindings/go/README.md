# Go bindings for transcribe.cpp

cgo bindings for the transcribe.cpp C API.

```go
s, err := transcribe.Open("moonshine-tiny-Q8_0.gguf", nil, nil)
if err != nil {
    log.Fatal(err)
}
defer s.Close()

res, err := s.Run(ctx, pcm, nil)   // mono float32, 16 kHz, [-1, 1]
if err != nil {
    log.Fatal(err)
}
fmt.Println(res.Text)
```

## Linking

cgo has no build step of its own, so it does not build the C++ tree the way
the Rust crate's `build.rs` does. Build and install libtranscribe first, then
point cgo at the prefix:

```
$ cmake -S . -B build -DTRANSCRIBE_BUILD_SHARED=ON -DCMAKE_BUILD_TYPE=Release
$ cmake --build build -j
$ cmake --install build --prefix "$PREFIX"

$ export CGO_CFLAGS="-I$PREFIX/include"
$ export CGO_LDFLAGS="-L$PREFIX/lib64"
$ export LD_LIBRARY_PATH="$PREFIX/lib64"
$ go build ./...
```

`lib64` is what the install writes on 64-bit Linux; other platforms use
`lib`. The install also drops a `transcribe-link.json` next to the library
listing exactly what to link, which is worth reading if a packaging step
needs to derive these flags rather than hard-code them.

## API

Handles are freed by `Close`, not by a finalizer, because a `Model` must
outlive every `Session` made from it and the garbage collector does not know
that ordering. `Open` bundles the two for the single-stream case, and its
`Close` frees both. `Session.Model` hands back a borrowed model that the
session owns, so closing that one is a no-op.

A `Result` owns everything in it. The library's own accessors return borrowed
pointers into session storage that the next run overwrites, so `Run` copies
the whole result out before returning; what you get back stays valid after
the next run and after the session is closed. `RunBatch` returns one
`BatchResult` per utterance, each with its own `Err`, since a batch can
succeed as a whole with one utterance failing inside it.

Cancelling the context passed to `Run` aborts between decode steps, so an
abort lands within tens of milliseconds of the request. The encoder cannot be
interrupted, so a short utterance whose cost is nearly all encoder will
finish anyway.

The rest of the contracts are on the API itself, so `go doc` is the place to
read them rather than here.

The first `LoadModel` or `Open` checks that the library it linked is the one
this binding was built for, in two ways. `CompiledVersion` must match the
library's base version, since pre-1.0 the ABI may break between minor
releases. `ABICheck` then compares every public struct's size and alignment
against what the library reports. The two catch different things: a
renumbered enum keeps every struct size, so only the version check sees it,
and a struct grown at the end within one release is only visible to
`ABICheck`. Neither runs more than once per process.

## Threading

A `Session` is single-threaded: never call two of its methods at once.

Concurrent *compute* is a 0.x limitation of the library rather than of this
binding, and it is wider than one session: at most one run, batch or active
stream may be in flight across **all** sessions of a given model. Sessions
share the model's backend instances, so overlapping runs corrupt decodes on
CPU and fail command buffers on Metal.

So to transcribe on several cores, load one `Model` per worker. Sharing one
model across a pool of sessions is supported only if you serialize the runs
yourself; this binding does not do it for you, unlike the Rust and TypeScript
bindings, which hold a model-wide lock. Everything else on a model is
thread-safe: capabilities, metadata, feature probes and `NewSession`.

## Streaming

`StreamBegin`, `StreamFeed` and `StreamFinalize` drive a live capture, on
models whose `Capabilities.SupportsStreaming` is set.

`StreamText` returns three views of the transcript. `Full` is the model's
current hypothesis and the authoritative one, but any of it can be rewritten
on the next feed. `Committed` is append-only, so it never flickers, at the
cost of being best-effort: a model that re-attends over a growing context can
revise something already committed, and since `Committed` is not rolled back,
`Committed`+`Tentative` stops reconstructing `Full` when that happens.
`StreamOptions.AgreementN` trades commit latency against that risk.

Render `StreamText` per feed. `StreamSnapshot` materializes the whole
hypothesis as a `Result` for callers that want rows instead, so calling it on
every feed makes the stream quadratic in the transcript.

Cancelling the context passed to `StreamBegin` is terminal for the stream: it
moves to `StreamFailed` and `StreamLastStatus` keeps `ErrAborted`.

## Family extensions

Knobs that only one architecture has ride `RunOptions.Family` and
`StreamOptions.Family`: `WhisperRunOptions`, `SortformerStreamOptions`,
`ParakeetStreamOptions`, `ParakeetBufferedStreamOptions`,
`MoonshineStreamingOptions` and `VoxtralRealtimeStreamOptions`.

A kind is legal in exactly one slot, and the Go types enforce that at
compile time: a stream extension does not satisfy `RunExtension`, so it
cannot be assigned to the run slot at all. Whether a *model* accepts a kind
is a runtime question, since it varies by variant rather than by family, and
`Model.AcceptsExtension` is the probe. Passing one a model does not take is
`ErrInvalidArg` rather than a silent no-op.

Their fields are pointers, where nil means the family's own default:

```go
res, err := s.Run(ctx, pcm, &transcribe.RunOptions{
    Family: &transcribe.WhisperRunOptions{
        InitialPrompt: "A presidential inaugural address.",
        Temperature:   transcribe.Opt[float32](0),
    },
})
```

The families disagree about which values are sentinels and several take a
meaningful zero, so a pointer is the only encoding that separates "leave it
alone" from "set it to zero" for every field. `Opt` wraps a literal into one.

## Not covered yet

`transcribe_ext_check`, which validates an extension's shape before a family
casts it; this binding builds the structs itself, so there is nothing for it
to check. The whisper chunk trace is also unbound, matching the Rust binding.

Windows is untested. cgo needs a GCC toolchain there, a different link
posture from the MSVC build the rest of this repo's Windows lanes use.

## Examples

The five canonical examples take a model and a clip from the command line,
fall back to the `TRANSCRIBE_SMOKE_*` variables CI exports, and skip cleanly
when neither is set:

```
$ go run ./examples/transcribe-file model.gguf audio.wav
$ go run ./examples/streaming streaming-model.gguf
$ go run ./examples/batch model.gguf
$ go run ./examples/backend-select          # device discovery needs no model
$ go run ./examples/error-handling model.gguf
```

## Tests

The tests that need a model skip unless one is named, so that nothing
downloads during a test run. Streaming takes its own variable, since not
every family streams:

```
$ go test ./...
$ TRANSCRIBE_SMOKE_MODEL=path/to/whisper-base.en-Q5_K_M.gguf \
    TRANSCRIBE_SMOKE_STREAMING_MODEL=path/to/moonshine-streaming-tiny-Q8_0.gguf \
    go test ./...
```

Run them against more than one family. Which result rows a model populates
and whether it can tokenize both vary by family, so a single model leaves
parts of the binding untested.

They transcribe `samples/jfk.wav`, found relative to this checkout rather
than to the working directory. Point `TRANSCRIBE_SMOKE_AUDIO` at a mono
16-bit 16 kHz clip to use another.

cgo's pointer rules are worth checking, since violating them fails rarely and
unpredictably rather than every time:

```
$ GOEXPERIMENT=cgocheck2 go test ./...
```
