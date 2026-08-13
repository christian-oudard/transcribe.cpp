package transcribe

// Public-ABI drift gate. cgo compiles include/transcribe.h directly, so there
// is no generated FFI layer to regenerate, and anything that stops compiling
// is caught by the build. What the build does not catch is an ABI change that
// still compiles: a renumbered enum, a reordered struct, a status value that
// moves. This binding aliases C enum values into Go constants and indexes
// positional tables with them, so those are exactly the changes that would
// break it silently.
//
// The gate is therefore this pinned hash, checked in CI against the neutral
// include/transcribe.abihash by scripts/ci/abihash_check.py, the same posture
// the Swift binding uses. When the header's ABI changes the neutral hash
// moves, the check goes red, and a maintainer bumps this constant after
// consciously reviewing what changed and auditing the binding for new or
// changed structs, enums and entry points.
//
// The per-field struct-layout check is waived, as it is for Swift and Rust,
// because cgo gets layout from a real compiler. ABICheck covers the other
// half at runtime: a header and a shared library that have drifted apart.
const PinnedHeaderHash = "7896d8d4c2a46147"

// HeaderHash is the public-ABI digest this binding was reviewed against.
func HeaderHash() string { return PinnedHeaderHash }
