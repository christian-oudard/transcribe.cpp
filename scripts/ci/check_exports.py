#!/usr/bin/env python3
"""check_exports.py - every TRANSCRIBE_API function is in the shared library.

TRANSCRIBE_API is a visibility attribute on a *declaration*. A definition does
not always inherit it: put one inside `namespace foo {` and it keeps the
translation unit's default, which this tree sets to hidden. The symbol then
exists with C linkage and is not exported, which a static archive resolves
happily and a shared library does not have at all.

So the failure is invisible in every build that links the archive -- the CLI,
the tests, the bench, and any consumer building from source -- and shows up
only when something links libtranscribe.so. That happened once, to the
titanet extension's init, and the first report of it was a downstream
project's undefined reference.

    check_exports.py --lib build/src/libtranscribe.so --include include

Usage note: run it against the built library, not the installed one. The two
are the same file, and the build tree is where it exists first.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# `TRANSCRIBE_API void transcribe_whatever_init(struct ... )`, with the return
# type and any newline between the macro and the name. Only functions: a
# declaration with no parenthesis is data, which this does not check.
DECL = re.compile(
    r"TRANSCRIBE_API\s+(?:[A-Za-z_][\w:]*\s+|\*\s*)+?(transcribe_\w+)\s*\(",
    re.MULTILINE,
)


def declared(include_dir: Path) -> dict[str, list[str]]:
    """Every TRANSCRIBE_API function name, and which header declares it."""
    out: dict[str, list[str]] = {}
    for header in sorted(include_dir.rglob("*.h")):
        for name in DECL.findall(header.read_text(errors="replace")):
            out.setdefault(name, []).append(str(header))
    return out


def exported(lib: Path) -> set[str]:
    """Every name the shared library defines in its dynamic symbol table."""
    try:
        raw = subprocess.run(
            ["nm", "-D", "--defined-only", str(lib)],
            check=True, capture_output=True, text=True,
        ).stdout
    except FileNotFoundError:
        sys.exit("nm is not on PATH; this check needs binutils")
    except subprocess.CalledProcessError as err:
        sys.exit(f"nm failed on {lib}: {err.stderr.strip()}")
    names = set()
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            names.add(parts[2])
    return names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lib", required=True, type=Path, help="built libtranscribe.so")
    ap.add_argument("--include", required=True, type=Path, help="public header directory")
    args = ap.parse_args()

    if not args.lib.exists():
        sys.exit(f"{args.lib} does not exist; build with -DTRANSCRIBE_BUILD_SHARED=ON")

    want = declared(args.include)
    have = exported(args.lib)
    missing = sorted(name for name in want if name not in have)

    print(f"{len(want)} functions declared TRANSCRIBE_API, {len(want) - len(missing)} exported")
    if missing:
        print("\nDeclared public and not exported:", file=sys.stderr)
        for name in missing:
            print(f"  {name}  ({', '.join(want[name])})", file=sys.stderr)
        print(
            "\nA definition inside a namespace does not inherit the visibility "
            "attribute from its declaration. Define it at global scope, or put "
            "TRANSCRIBE_API on the definition too.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
