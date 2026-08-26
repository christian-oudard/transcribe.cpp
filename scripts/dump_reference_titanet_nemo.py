#!/usr/bin/env python3
"""
dump_reference_titanet_nemo.py - TitaNet speaker embedding reference tensors
from NVIDIA NeMo (the canonical reference).

TitaNet is a speaker embedding model (arch pattern `encoder-embedding`), NOT a
transcription model. There is no tokenizer and no text output: the reference
behavioural artifact is one 192-dim vector per clip, and the thing that has to
match is not a string but a direction in that space.

Pipeline (whole-clip forward):
    preprocessor      mel, 80 x T_mel, per-feature normalized
    -> encoder        5 ContextNet blocks, separable convs + squeeze-excite,
                      80 -> 1024 -> 3072 channels, no time subsampling
    -> pooling        attentive statistics: softmax over time, weighted mean
                      and standard deviation concatenated -> 6144
    -> emb_layers     BatchNorm1d + Linear 6144 -> 192

Two reference paths are dumped:
  * `encoder`   runs the forward pass with hooks and dumps the mel, the
                per-block encoder activations, the pooled statistics and the
                final embedding. This is what the C++ graph must match stage
                by stage.
  * `verify`    embeds a pair of clips and writes their cosine similarity,
                which is the behavioural artifact: the model is used by asking
                whether two vectors are close, never by reading one absolutely.

Every stage here is whole-clip. Squeeze-excite averages over the entire clip
at se_context_size -1 and attentive pooling takes a softmax and a standard
deviation across it, so no activation in this model can be reproduced from a
window of the audio. Dumping a prefix and comparing it to the same prefix of a
longer dump will not match, by construction.

Tensor output uses the shared contract via scripts.lib.ref_dump
(write_tensor records rms / p99_abs for magnitude-aware tolerances).

    uv run --project scripts/envs/titanet \
      scripts/dump_reference_titanet_nemo.py encoder \
      --model nvidia/speakerverification_en_titanet_large \
      --audio samples/jfk.wav \
      --out build/validate/titanet/speakerverification_en_titanet_large/jfk/encoder/ref

    uv run --project scripts/envs/titanet \
      scripts/dump_reference_titanet_nemo.py verify \
      --model nvidia/speakerverification_en_titanet_large \
      --audio samples/jfk.wav --audio-b samples/dots.wav \
      --out build/validate/titanet/speakerverification_en_titanet_large/jfk/verify/ref
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import ref_dump  # noqa: E402

write_tensor = ref_dump.write_tensor


def _load(model: str):
    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    if model.endswith(".nemo") or Path(model).exists():
        m = EncDecSpeakerLabelModel.restore_from(
            restore_path=model, map_location="cpu", strict=False
        )
    else:
        m = EncDecSpeakerLabelModel.from_pretrained(model, map_location="cpu")
    m.eval()
    return m


def _read_audio(path: str) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sr)


def _time_major(a: np.ndarray) -> np.ndarray:
    """Squeeze batch and return [T, C].

    The layout is stated rather than guessed. ConvASR is channel-major
    throughout, [B, C, T], because it is convolutional rather than
    attentional, so the transpose is unconditional. Inferring it from which
    axis is longer works for the four 1024-wide blocks on an eleven second
    clip and silently fails on the 3072-wide one, which has more channels
    than frames.
    """
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2:
        raise ValueError(f"expected [B, C, T] or [C, T], got {a.shape}")
    return np.ascontiguousarray(a.T.astype(np.float32))


def _src(model: str, hook: str) -> dict[str, Any]:
    return {"framework": "nemo", "model": model, "hook": hook}


def _embed(m, audio: np.ndarray, sub_block: int | None = None) -> tuple[np.ndarray, dict[str, torch.Tensor]]:
    sig = torch.tensor(audio).unsqueeze(0)
    length = torch.tensor([audio.shape[0]])
    captured: dict[str, torch.Tensor] = {}

    def grab(name):
        def hook(_mod, _inp, out):
            # A JasperBlock returns ([activations], lengths), so the tensor is
            # two containers deep, while the encoder and the pooling return it
            # directly. Unwrap until there is a tensor rather than assuming a
            # depth per module.
            t = out
            while isinstance(t, (tuple, list)):
                t = t[0]
            captured[name] = t.detach().to(torch.float32).cpu()

        return hook

    handles = [m.encoder.register_forward_hook(grab("encoder"))]
    for i, block in enumerate(m.encoder.encoder):
        handles.append(block.register_forward_hook(grab(f"block.{i}")))
    handles.append(m.decoder._pooling.register_forward_hook(grab("pooling")))
    # Inside one block, for bisecting. A block is a loop of identical-looking
    # sub-layers, so a divergence that first shows at the block output says
    # only "somewhere in here"; these say which repeat, and whether it was the
    # squeeze-excite or the skip path.
    if sub_block is not None:
        blk = m.encoder.encoder[sub_block]
        for i, layer in enumerate(blk.mconv):
            handles.append(layer.register_forward_hook(grab(f"sub.mconv.{i}")))
        if getattr(blk, "res", None) is not None:
            for i, layer in enumerate(blk.res[0]):
                handles.append(layer.register_forward_hook(grab(f"sub.res.{i}")))
    try:
        with torch.no_grad():
            _logits, emb = m.forward(input_signal=sig, input_signal_length=length)
    finally:
        for h in handles:
            h.remove()
    return emb.detach().cpu().numpy()[0].astype(np.float32), captured


def cmd_encoder(args: argparse.Namespace) -> int:
    m = _load(args.model)
    audio, _sr = _read_audio(args.audio)
    out_dir = Path(args.out)

    sig = torch.tensor(audio).unsqueeze(0)
    length = torch.tensor([audio.shape[0]])
    with torch.no_grad():
        mel, _mel_len = m.preprocessor(input_signal=sig, length=length)
    write_tensor(
        "enc.mel.in",
        np.ascontiguousarray(mel[0].numpy().astype(np.float32)),
        "frontend",
        _src(args.model, "preprocessor.out"),
        out_dir=out_dir,
    )

    emb, captured = _embed(m, audio, args.sub_block)

    # Per-block outputs, because the blocks differ in a way that matters: the
    # three middle ones carry a residual and the outer two do not, so a bug in
    # the skip path shows up here and nowhere earlier.
    for name, tensor in sorted(captured.items()):
        if not name.startswith("block."):
            continue
        i = name.split(".")[1]
        write_tensor(
            f"enc.block.{i}.out",
            _time_major(tensor.numpy()),
            "encoder",
            _src(args.model, f"encoder.encoder.{i}.out"),
            out_dir=out_dir,
        )
    write_tensor(
        "enc.out",
        _time_major(captured["encoder"].numpy()),
        "encoder",
        _src(args.model, "encoder.out"),
        out_dir=out_dir,
    )
    # The pooled statistics are where time disappears. Everything before this
    # is [T, C] and everything after is one vector, so a length-handling bug
    # lands exactly here.
    write_tensor(
        "pool.out",
        np.ascontiguousarray(captured["pooling"].numpy().reshape(-1).astype(np.float32)),
        "pooling",
        _src(args.model, "decoder._pooling.out"),
        out_dir=out_dir,
    )
    write_tensor(
        "emb.out",
        emb,
        "embedding",
        _src(args.model, "forward.emb"),
        out_dir=out_dir,
    )
    # Debug tensors, not gate tensors: they exist to bisect a divergence and
    # carry no tolerance entry.
    for name, tensor in sorted(captured.items()):
        if not name.startswith("sub."):
            continue
        write_tensor(
            f"enc.block.{args.sub_block}.{name[4:]}",
            _time_major(tensor.numpy()),
            "encoder",
            _src(args.model, f"encoder.encoder.{args.sub_block}.{name[4:]}"),
            out_dir=out_dir,
        )
    print(f"wrote encoder-stage tensors to {out_dir}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    if not args.audio_b:
        raise SystemExit("error: verify needs --audio-b, since a similarity takes two clips")
    m = _load(args.model)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    a, _ = _read_audio(args.audio)
    b, _ = _read_audio(args.audio_b)
    ea, _ = _embed(m, a)
    eb, _ = _embed(m, b)
    cosine = float(
        np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb))
    )

    write_tensor("emb.a", ea, "embedding", _src(args.model, "forward.emb(a)"), out_dir=out_dir)
    write_tensor("emb.b", eb, "embedding", _src(args.model, "forward.emb(b)"), out_dir=out_dir)
    # The scalar the model is actually used for. A port that reproduces both
    # vectors but not this is a port that got the norms right and the
    # directions wrong, which no per-tensor tolerance would catch.
    (out_dir / "verify.json").write_text(
        json.dumps(
            {
                "audio_a": args.audio,
                "audio_b": args.audio_b,
                "cosine": cosine,
                "embedding_size": int(ea.shape[0]),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"cosine({Path(args.audio).name}, {Path(args.audio_b).name}) = {cosine:.6f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("encoder", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True, help="HF repo id or path to a .nemo")
        p.add_argument("--audio", required=True)
        p.add_argument("--audio-b", default=None, help="second clip, for verify")
        p.add_argument("--out", required=True)
        p.add_argument(
            "--sub-block",
            type=int,
            default=None,
            help="also dump every sub-layer output inside this encoder block",
        )
    args = ap.parse_args()
    return {"encoder": cmd_encoder, "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
