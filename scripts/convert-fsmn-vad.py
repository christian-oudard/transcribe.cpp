#!/usr/bin/env python3
"""
convert-fsmn-vad.py - convert FunASR's FSMN voice activity detector
(speech_fsmn_vad_zh-cn-16k-common) into a GGUF.

A voice activity detector answers one question per frame: is anybody speaking.
It is here because diarization needs it and nothing else can supply it. The
clustering counts an extra speaker on most meetings, and the extra cluster is
audible non-speech -- a chair, a keyboard -- that an energy threshold cannot
tell from a voice; see docs/diarization.md, which records five gates that were
measured and none that worked.

The whole model is 24 tensors and 1.7 MB:

    in_linear1     400 -> 140   affine
    in_linear2     140 -> 250   affine, then ReLU
    fsmn x4        250 -> 128   linear, no bias
                   depthwise conv over the last 20 frames, added to its input
                   128 -> 250   affine, then ReLU
    out_linear1    250 -> 140   affine
    out_linear2    140 -> 248   affine, then softmax

The 248 outputs are HMM state posteriors and only one of them is used: pdf 0 is
silence, so the probability of speech in a frame is 1 minus that. The rest are
what the model was trained against and say nothing anyone here wants.

The memory is left-only (lorder 20, rorder 0), which makes this causal: the
answer for a frame never depends on audio after it. That is what makes it
usable both for a stream and for a file, and it is why this model was chosen
over silero, which carries an LSTM state between calls.

The frontend is kaldi fbank with LFR stacking, which transcribe.cpp already
has for the other FunASR models; the CMVN statistics that normalize those
features live in a separate file (am.mvn) and are carried into the GGUF here,
since a model whose normalization is missing produces confident nonsense.

Usage:
    uv run --project scripts/envs/funasr scripts/convert-fsmn-vad.py \
      funasr/fsmn-vad --repo-id funasr/fsmn-vad
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGMLQuantizationType

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.gguf_common import (  # noqa: E402
    add_general_identity,
    gguf_writer,
)

# Per-layer tensors, in the order they run. The names on the left are what the
# checkpoint calls them; the ones on the right are what the C++ looks up.
LAYER_TABLE = [
    ("linear.linear.weight", "linear.weight"),
    ("fsmn_block.conv_left.weight", "memory.weight"),
    ("affine.linear.weight", "affine.weight"),
    ("affine.linear.bias", "affine.bias"),
]

HEAD_TABLE = [
    ("encoder.in_linear1.linear.weight", "in.0.weight"),
    ("encoder.in_linear1.linear.bias", "in.0.bias"),
    ("encoder.in_linear2.linear.weight", "in.1.weight"),
    ("encoder.in_linear2.linear.bias", "in.1.bias"),
    ("encoder.out_linear1.linear.weight", "out.0.weight"),
    ("encoder.out_linear1.linear.bias", "out.0.bias"),
    ("encoder.out_linear2.linear.weight", "out.1.weight"),
    ("encoder.out_linear2.linear.bias", "out.1.bias"),
]


def read_cmvn(path: Path, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """The additive shift and multiplicative scale from a kaldi-style am.mvn.

    Already in the form the features are normalized with: x * scale + shift.

    The vectors are inside brackets, and only inside brackets. The lines that
    carry them read `<LearnRateCoef> 0 [ -8.31 -8.60 ... ]`, so taking every
    number on the line picks up the coefficient as well and shifts all 400
    values one place along -- which does not fail, it normalizes every feature
    by its neighbour's statistics. The length check below is what would catch
    it if this parse ever went wrong again.
    """
    found = re.findall(r"\[([^\]]*)\]", path.read_text())
    vectors = []
    for body in found:
        numbers = [float(v) for v in body.split()]
        if len(numbers) == dim:
            vectors.append(np.array(numbers, dtype=np.float32))
    if len(vectors) != 2:
        raise SystemExit(f"{path}: expected two vectors of {dim}, found {len(vectors)}")
    return vectors[0], vectors[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", help="directory holding model.pt, config.yaml and am.mvn")
    ap.add_argument("--repo-id", default="funasr/fsmn-vad")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = Path(args.source)
    state = torch.load(src / "model.pt", map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    shift, scale = read_cmvn(src / "am.mvn", 400)

    layers = sorted({int(m.group(1)) for k in state if (m := re.match(r"encoder\.fsmn\.(\d+)\.", k))})
    if not layers or layers != list(range(len(layers))):
        raise SystemExit(f"unexpected fsmn layers: {layers}")

    out_path = Path(args.out) if args.out else Path("models/fsmn-vad/fsmn-vad-F32.gguf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf_writer(str(out_path), "fsmn-vad")
    add_general_identity(
        writer,
        name="FSMN Voice Activity Detection",
        basename="fsmn-vad",
        repo_url=f"https://huggingface.co/{args.repo_id}",
        description=(
            "Per-frame speech probability. Four FSMN layers over kaldi fbank "
            "with LFR stacking; causal, so a frame's answer never depends on "
            "audio after it."
        ),
        languages=["en"],
    )

    # The frontend, which the C++ has already: 80 mels, 25 ms window, 10 ms
    # hop, hamming, then five frames stacked with no downsampling.
    writer.add_uint32("stt.frontend.sample_rate", 16000)
    writer.add_uint32("stt.frontend.num_mels", 80)
    writer.add_uint32("stt.frontend.win_length", 400)
    writer.add_uint32("stt.frontend.hop_length", 160)
    writer.add_string("stt.frontend.window", "hamming")
    writer.add_uint32("stt.frontend.lfr_m", 5)
    writer.add_uint32("stt.frontend.lfr_n", 1)

    writer.add_uint32("stt.fsmn_vad.layers", len(layers))
    writer.add_uint32("stt.fsmn_vad.input_dim", 400)
    writer.add_uint32("stt.fsmn_vad.linear_dim", 250)
    writer.add_uint32("stt.fsmn_vad.proj_dim", 128)
    writer.add_uint32("stt.fsmn_vad.lorder", 20)
    writer.add_uint32("stt.fsmn_vad.output_dim", 248)
    # Which of the 248 outputs means silence. Everything else is speech, and
    # no caller should ever read an individual one.
    writer.add_uint32("stt.fsmn_vad.silence_pdf", 0)

    writer.add_bool("stt.capability.speech_activity", True)
    for key in ("streaming", "speaker_diarization", "lang_detect", "translate", "timestamps"):
        writer.add_bool(f"stt.capability.{key}", False)

    emitted = 0

    def emit(name: str, tensor: torch.Tensor) -> None:
        nonlocal emitted
        array = tensor.detach().to(torch.float32).numpy()
        writer.add_tensor(name, np.ascontiguousarray(array), raw_dtype=GGMLQuantizationType.F32)
        emitted += 1

    # Named the way sensevoice names them, which is what the quantizer's
    # policy looks for: normalization statistics must stay F32, and a name it
    # does not recognize is a name it quantizes.
    emit("frontend.cmvn.shift", torch.from_numpy(shift))
    emit("frontend.cmvn.scale", torch.from_numpy(scale))

    for src_name, dst in HEAD_TABLE:
        emit(dst, state[src_name])

    for i in layers:
        for src_suffix, dst_suffix in LAYER_TABLE:
            tensor = state[f"encoder.fsmn.{i}.{src_suffix}"]
            if dst_suffix == "memory.weight":
                # [C, 1, lorder, 1] in the checkpoint, because FunASR runs it
                # as a 2-D convolution over a degenerate axis. Squeeze it to
                # [C, lorder]: a depthwise convolution over time is what it is.
                tensor = tensor.squeeze(-1).squeeze(1)
            emit(f"fsmn.{i}.{dst_suffix}", tensor)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"{len(layers)} layers, {emitted} tensors")
    print(f"Wrote GGUF: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
