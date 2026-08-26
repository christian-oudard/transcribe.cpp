# Third-party licenses

transcribe.cpp is MIT-licensed (see [`LICENSE`](LICENSE)). It vendors and links
the third-party components below, each under its own permissive license. The
authoritative license text for each ships in-tree at the path noted; copies are
reproduced here for convenience.

| Component | License | Vendored at | Pin |
|-----------|---------|-------------|-----|
| ggml      | MIT     | [`ggml/LICENSE`](ggml/LICENSE) | see [`ggml/UPSTREAM`](ggml/UPSTREAM) |
| miniz     | MIT     | [`src/third_party/miniz/LICENSE`](src/third_party/miniz/LICENSE) | see [`src/third_party/miniz/UPSTREAM`](src/third_party/miniz/UPSTREAM) |
| fastcluster | BSD-2-Clause | [`src/third_party/fastcluster/LICENSE`](src/third_party/fastcluster/LICENSE) | see [`src/third_party/fastcluster/UPSTREAM`](src/third_party/fastcluster/UPSTREAM) |

Prebuilt artifacts (the Swift xcframework, the native Python wheels, and the
npm platform packages) carry these same texts alongside the binaries — see each
binding's packaging for the bundled `LICENSE.ggml` / `LICENSE.miniz` files.

---

## ggml

The tensor library underlying transcribe.cpp. Vendored under `ggml/`.

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## miniz

A single-file deflate/inflate (zlib-subset) codec. transcribe.cpp uses it only
for Whisper's temperature-fallback compression-ratio heuristic; it replaces the
previous system-zlib dependency. Vendored under `src/third_party/miniz/`.

miniz is MIT-licensed. (The comment block at the top of the amalgamated
`miniz.c` / `miniz.h` still reads "public domain" — a stale artifact from the
project's pre-relicensing history; upstream's current and authoritative license
is the MIT text below, also shipped at `src/third_party/miniz/LICENSE`.)

```
Copyright 2013-2014 RAD Game Tools and Valve Software
Copyright 2010-2014 Rich Geldreich and Tenacious Software LLC

All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

---

## fastcluster

Fast hierarchical agglomerative clustering: Daniel Müllner's implementation,
through Christoph Dalitz's standalone C++ interface. transcribe.cpp uses it for
the last stage of speaker diarization, grouping the embedding of each window of
speech by whose voice it is. Vendored under `src/third_party/fastcluster/`.

BSD 2-clause.

```
Copyright:
  * fastcluster_dm.cpp & fastcluster_R_dm.cpp:
     © 2011 Daniel Müllner <http://danifold.net>
  * fastcluster.(h|cpp) & demo.cpp & plotresult.r:
     © 2018 Christoph Dalitz <http://www.hsnr.de/ipattern/>
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
