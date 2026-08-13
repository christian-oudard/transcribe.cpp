#!/usr/bin/env python3
"""Fail if a binding's pinned public-ABI hash drifts from the header.

Bindings that generate an FFI layer are drift-gated by regenerating it
(notes/bindings-requirements.md §2). The ones whose toolchain reads the C
headers directly have nothing to regenerate -- Swift's Clang importer and Go's
cgo both compile the header on every build -- so their gate is a PINNED
constant, checked here against the neutral ``include/transcribe.abihash``
emitted by the Python generator (the hash oracle).

Compiling the header catches only ABI changes that stop compiling. A renumbered
enum, a reordered struct, or a moved status value compiles clean and breaks
these bindings silently, which is what the pin is for. When the header's ABI
changes the neutral hash moves, this check goes red, and a maintainer bumps the
pinned constant after consciously reviewing the change and auditing the wrapper.

    uv run --no-project scripts/ci/abihash_check.py [binding ...]

With no arguments every binding below is checked. Exit 0 when they agree; 1 on
drift; 2 if either value could not be located.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ABIHASH_FILE = REPO / "include" / "transcribe.abihash"

# name -> (file holding the pin, regex whose first group is the hash)
PINNED: dict[str, tuple[Path, str]] = {
    "swift": (
        REPO / "bindings" / "swift" / "Sources" / "TranscribeCpp" / "ABIHash.swift",
        r'pinnedHeaderHash\s*=\s*"([0-9a-fA-F]+)"',
    ),
    "go": (
        REPO / "bindings" / "go" / "abihash.go",
        r'PinnedHeaderHash\s*=\s*"([0-9a-fA-F]+)"',
    ),
}


def check(name: str, neutral: str) -> int:
    pin_file, pattern = PINNED[name]
    if not pin_file.exists():
        print(f"error: missing {pin_file}", file=sys.stderr)
        return 2
    m = re.search(pattern, pin_file.read_text())
    if not m:
        print(f"error: could not find the pinned hash in {pin_file}", file=sys.stderr)
        return 2
    pinned = m.group(1)

    if pinned != neutral:
        print(
            f"{name} ABI-hash drift: the public header ABI changed.\n"
            f"  include/transcribe.abihash : {neutral}\n"
            f"  {pin_file.name} (pinned)   : {pinned}\n"
            "Review the header change, audit the wrapper for new/changed structs,"
            " enums, or entry points, then update the pinned constant.",
            file=sys.stderr,
        )
        return 1

    print(f"{name} abihash ok: {neutral}")
    return 0


def main(argv: list[str]) -> int:
    if not ABIHASH_FILE.exists():
        print(f"error: missing {ABIHASH_FILE}", file=sys.stderr)
        return 2
    neutral = ABIHASH_FILE.read_text().strip()

    names = argv or sorted(PINNED)
    unknown = [n for n in names if n not in PINNED]
    if unknown:
        print(f"error: unknown binding(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    return max(check(n, neutral) for n in names)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
