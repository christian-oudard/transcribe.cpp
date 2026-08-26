#!/usr/bin/env python3
"""
dump_reference_fsmn_vad_funasr.py - reference tensors for FunASR's FSMN voice
activity detector.

The model answers one question per frame -- is anybody speaking -- and the
whole of it is four FSMN layers between two pairs of affine transforms. What
has to match is therefore short: the features going in, each layer's output,
and the speech probability coming out.

The behavioural artifact is not a transcript but the regions: the stretches of
audio the detector calls speech. That is what diarization consumes, and a port
that reproduces every tensor and disagrees about the regions has not worked.

Usage:
    uv run --project scripts/envs/funasr_nano \
      scripts/dump_reference_fsmn_vad_funasr.py \
      --model ~/vad-src --audio samples/jfk.wav \
      --out build/validate/fsmn_vad/fsmn-vad/jfk/ref
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.lib.ref_dump import write_tensor  # noqa: E402


def src(model: str, hook: str) -> dict:
    return {"framework": "funasr", "model": model, "hook": hook}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="directory with model.pt, config.yaml, am.mvn")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out)

    from funasr import AutoModel

    auto = AutoModel(model=args.model, disable_update=True, device="cpu")
    model = auto.model
    frontend = auto.kwargs["frontend"]

    import soundfile

    audio, rate = soundfile.read(args.audio, dtype="float32")
    if rate != 16000:
        raise SystemExit(f"{args.audio} is {rate} Hz; the model takes 16 kHz")

    # The frontend is kaldi fbank plus LFR stacking plus CMVN, which is what
    # transcribe.cpp already has for the other FunASR models. Dumping its
    # output separates a frontend disagreement from a model one, which is the
    # first thing to know when the probabilities differ.
    feats, _lengths = frontend(torch.from_numpy(audio)[None, :], [len(audio)])
    write_tensor("enc.feats.in", np.ascontiguousarray(feats[0].numpy()), "frontend",
                 src(args.model, "frontend.out"), out_dir=out_dir)

    captured: dict[str, torch.Tensor] = {}

    def grab(name):
        def hook(_mod, _inp, out):
            tensor = out[0] if isinstance(out, (tuple, list)) else out
            captured[name] = tensor.detach().to(torch.float32).cpu()

        return hook

    encoder = model.encoder
    handles = [
        encoder.in_linear1.register_forward_hook(grab("in.0")),
        encoder.in_linear2.register_forward_hook(grab("in.1")),
        encoder.out_linear1.register_forward_hook(grab("out.0")),
        encoder.out_linear2.register_forward_hook(grab("out.1")),
    ]
    for i, block in enumerate(encoder.fsmn):
        handles.append(block.register_forward_hook(grab(f"fsmn.{i}")))

    try:
        with torch.no_grad():
            scores = encoder(feats, in_cache=None) if _takes_cache(encoder) else encoder(feats)
    finally:
        for h in handles:
            h.remove()

    if isinstance(scores, (tuple, list)):
        scores = scores[0]
    scores = scores.detach().to(torch.float32).cpu()

    for name, tensor in sorted(captured.items()):
        write_tensor(f"enc.{name}.out", np.ascontiguousarray(tensor[0].numpy()), "encoder",
                     src(args.model, name), out_dir=out_dir)

    # The posteriors over 248 HMM states, and the one number anyone reads: the
    # probability that a frame is speech, which is one minus the silence state.
    probs = np.ascontiguousarray(scores[0].numpy())
    write_tensor("vad.posteriors", probs, "head", src(args.model, "encoder.out"), out_dir=out_dir)
    speech = 1.0 - probs[:, 0]
    write_tensor("vad.speech", np.ascontiguousarray(speech), "head",
                 src(args.model, "1 - posteriors[:, silence]"), out_dir=out_dir)

    # The artifact the port is judged on: how much of the clip is speech, and
    # where. A port that matches every tensor and disagrees here has not worked.
    frames = int(speech.shape[0])
    (out_dir / "speech.json").write_text(
        json.dumps(
            {
                "audio": args.audio,
                "frames": frames,
                "frame_ms": 10,
                "speech_frames": int((speech > 0.5).sum()),
                "speech_fraction": float((speech > 0.5).mean()),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"{frames} frames, {(speech > 0.5).mean():.1%} speech, written to {out_dir}")
    return 0


def _takes_cache(encoder) -> bool:
    import inspect

    return "in_cache" in inspect.signature(encoder.forward).parameters


if __name__ == "__main__":
    raise SystemExit(main())
