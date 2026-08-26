#!/usr/bin/env python3
"""
convert-titanet.py - convert NVIDIA TitaNet speaker embedding models
(.nemo, NeMo EncDecSpeakerLabelModel) into a reference-dtype GGUF.

TitaNet is an `encoder-embedding`: no tokenizer, no decoder, no text, and no
timestamps. One run takes audio and returns a single fixed-length vector in
which two clips of the same voice land close together. That is the piece the
library is missing for diarization: sortformer tracks speakers but caps at
four and produces no embeddings, so there is nothing to cluster over a whole
recording. With vectors, the speaker count comes out of the data.

Pipeline and tensor sources (124 tensors in the .nemo state_dict):

    preprocessor.featurizer.*  -> skipped (C++ recomputes the mel frontend)
    encoder.encoder.{b}.mconv.*  -> enc.blocks.{b}.rep.{r}.*  (ContextNet blocks)
    encoder.encoder.{b}.mconv.*.fc.*  -> enc.blocks.{b}.se.*  (squeeze-excite)
    encoder.encoder.{b}.res.0.*  -> enc.blocks.{b}.res.*      (pointwise + BN)
    decoder._pooling.attention_layer.*  -> pool.*             (attentive stats)
    decoder.emb_layers.0.*     -> emb.*                       (BN + Linear -> 192)
    decoder.final.*            -> skipped (16681-way angular softmax, training only)

Each block repeats a [depthwise conv, pointwise conv, batch norm, relu,
dropout] group `repeat` times, so the mconv indices step by 5 and the
squeeze-excite module sits at 5*repeat-2. Blocks with `residual: true` carry a
pointwise conv and batch norm on the skip path.

The final 16681-class layer classifies the training speakers and says nothing
about anyone else's voice, so inference stops at the 192-dim embedding.

Reference dtype is F32 (the NeMo speaker-model state_dict is fp32).

Usage (via the TitaNet reference env, which has NeMo):
    uv run --project scripts/envs/titanet \
      scripts/convert-titanet.py nvidia/speakerverification_en_titanet_large \
      --repo-id nvidia/speakerverification_en_titanet_large
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.gguf_common import (  # noqa: E402
    add_general_identity,
    canonicalize_normalize,
    encode_for_gguf,
    gguf_name,
    gguf_writer,
    reference_dtype_for,
    slug_from_repo_id,
)

REFERENCE_TYPE = GGMLQuantizationType.F32
REFERENCE_DTYPE_LABEL = "F32"
REFERENCE_FILE_TYPE = 0  # LlamaFileType ALL_F32

# One [depthwise, pointwise, batchnorm, relu, dropout] group per repeat. Only
# three of the five carry weights; the stride is what the other two cost.
MCONV_STRIDE = 5

# Suffixes within one repeat group, relative to its base index.
REPEAT_TABLE = [
    ("{i}.conv.weight", "dw.weight"),        # depthwise, groups=channels
    ("{j}.conv.weight", "pw.weight"),        # pointwise, 1x1
    ("{k}.weight", "bn.weight"),
    ("{k}.bias", "bn.bias"),
    ("{k}.running_mean", "bn.running_mean"),
    ("{k}.running_var", "bn.running_var"),
]

# Squeeze-excite: two biasless linears, C -> C/8 -> C, applied to the
# time-averaged channel vector. se_context_size is -1 in every published
# TitaNet, meaning the average is over the whole clip.
SE_TABLE = [
    ("fc.0.weight", "se.down.weight"),
    ("fc.2.weight", "se.up.weight"),
]

# Skip path on the residual blocks: pointwise conv then batch norm.
RES_TABLE = [
    ("res.0.0.conv.weight", "res.pw.weight"),
    ("res.0.1.weight", "res.bn.weight"),
    ("res.0.1.bias", "res.bn.bias"),
    ("res.0.1.running_mean", "res.bn.running_mean"),
    ("res.0.1.running_var", "res.bn.running_var"),
]

# Attentive statistics pooling, then the embedding layer. The attention is a
# conv-batchnorm-tanh-conv stack over time producing one logit per channel per
# frame; softmax over time gives weights for the mean and standard deviation,
# which are concatenated, so the embedding layer sees 2x the encoder width.
HEAD_TABLE = [
    ("decoder._pooling.attention_layer.0.conv_layer.weight", "pool.attn.0.weight"),
    ("decoder._pooling.attention_layer.0.conv_layer.bias", "pool.attn.0.bias"),
    ("decoder._pooling.attention_layer.0.bn.weight", "pool.attn.bn.weight"),
    ("decoder._pooling.attention_layer.0.bn.bias", "pool.attn.bn.bias"),
    ("decoder._pooling.attention_layer.0.bn.running_mean", "pool.attn.bn.running_mean"),
    ("decoder._pooling.attention_layer.0.bn.running_var", "pool.attn.bn.running_var"),
    ("decoder._pooling.attention_layer.2.weight", "pool.attn.1.weight"),
    ("decoder._pooling.attention_layer.2.bias", "pool.attn.1.bias"),
    ("decoder.emb_layers.0.0.weight", "emb.bn.weight"),
    ("decoder.emb_layers.0.0.bias", "emb.bn.bias"),
    ("decoder.emb_layers.0.0.running_mean", "emb.bn.running_mean"),
    ("decoder.emb_layers.0.0.running_var", "emb.bn.running_var"),
    ("decoder.emb_layers.0.1.weight", "emb.proj.weight"),
    ("decoder.emb_layers.0.1.bias", "emb.proj.bias"),
]

# The classifier over training speakers, and the frontend the C++ recomputes.
EXPECTED_UNUSED_PREFIXES = ("preprocessor.", "decoder.final.")
EXPECTED_UNUSED_SUFFIXES = (".num_batches_tracked",)


def _to_fp32(t) -> np.ndarray:
    import torch

    if not isinstance(t, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(t).__name__}")
    if t.dtype != torch.float32:
        raise ValueError(f"expected fp32 tensor, got {t.dtype} — cast at the source")
    return np.ascontiguousarray(t.detach().cpu().numpy())


def _add(writer, name: str, arr: np.ndarray) -> None:
    ggml_type = reference_dtype_for(name, REFERENCE_TYPE)
    data, out_type = encode_for_gguf(arr, ggml_type)
    writer.add_tensor(name, data, raw_dtype=out_type)


def convert(model_spec: str, out_path: Path, repo_id: str | None = None) -> None:
    from omegaconf import OmegaConf
    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    print(f"Output dtype: {REFERENCE_DTYPE_LABEL} (source/reference dtype)")
    if model_spec.endswith(".nemo") or Path(model_spec).exists():
        model = EncDecSpeakerLabelModel.restore_from(
            restore_path=model_spec, map_location="cpu", strict=False
        )
    else:
        model = EncDecSpeakerLabelModel.from_pretrained(model_spec, map_location="cpu")
    model.eval()
    cfg = OmegaConf.to_container(model.cfg, resolve=True)
    sd = model.state_dict()
    sd_keys = set(sd)

    enc = cfg["encoder"]
    dec = cfg["decoder"]
    pre = cfg["preprocessor"]
    blocks = enc["jasper"]
    emb_size = int(dec["emb_sizes"])
    feat_out = int(dec["feat_in"])
    print(f"blocks={len(blocks)} feat_out={feat_out} embedding={emb_size}")

    writer = gguf_writer(str(out_path), "titanet")

    add_general_identity(
        writer,
        name="TitaNet Large Speaker Verification",
        basename="titanet",
        size_label="25M",
        version="v1",
        file_type=REFERENCE_FILE_TYPE,
        languages=["en"],
        author="NVIDIA",
        organization="nvidia",
        license="cc-by-4.0",
        license_name="cc-by-4.0",
        license_link="https://creativecommons.org/licenses/by/4.0/",
        repo_url=f"https://huggingface.co/{repo_id}" if repo_id else None,
        description=(
            "Speaker embedding model (encoder-embedding): ContextNet encoder with "
            "squeeze-excite, attentive statistics pooling, one 192-dim vector per clip."
        ),
    )

    writer.add_string("stt.variant", out_path.parent.name)

    # ----- frontend (C++ recomputes; these are the reference params) -----
    sr = int(pre["sample_rate"])
    writer.add_uint32("stt.frontend.sample_rate", sr)
    writer.add_uint32("stt.frontend.num_mels", int(pre["features"]))
    writer.add_uint32("stt.frontend.n_fft", int(pre["n_fft"]))
    writer.add_uint32("stt.frontend.hop_length", int(round(float(pre["window_stride"]) * sr)))
    writer.add_uint32("stt.frontend.win_length", int(round(float(pre["window_size"]) * sr)))
    writer.add_string("stt.frontend.window", str(pre.get("window", "hann")))
    writer.add_string("stt.frontend.normalize", canonicalize_normalize(pre.get("normalize")))
    writer.add_float32(
        "stt.frontend.pre_emphasis",
        float(pre.get("preemph") if pre.get("preemph") is not None else 0.97),
    )
    writer.add_float32("stt.frontend.dither", float(pre.get("dither", 0.0)))

    # ----- capabilities -----
    # Everything a transcriber does, this does not do. It hears who rather
    # than what, so the one capability it claims is the embedding itself.
    writer.add_bool("stt.capability.speaker_embedding", True)
    writer.add_bool("stt.capability.streaming", False)
    writer.add_bool("stt.capability.speaker_diarization", False)
    writer.add_bool("stt.capability.lang_detect", False)
    writer.add_bool("stt.capability.translate", False)
    writer.add_bool("stt.capability.timestamps", False)

    # ----- architecture dims -----
    writer.add_uint32("stt.titanet.embedding_size", emb_size)
    writer.add_uint32("stt.titanet.encoder.feat_in", int(enc["feat_in"]))
    writer.add_uint32("stt.titanet.encoder.feat_out", feat_out)
    writer.add_uint32("stt.titanet.encoder.n_blocks", len(blocks))
    writer.add_string("stt.titanet.encoder.activation", str(enc.get("activation", "relu")))
    writer.add_array("stt.titanet.encoder.filters", [int(b["filters"]) for b in blocks])
    writer.add_array("stt.titanet.encoder.repeat", [int(b["repeat"]) for b in blocks])
    writer.add_array("stt.titanet.encoder.kernel", [int(b["kernel"][0]) for b in blocks])
    writer.add_array("stt.titanet.encoder.stride", [int(b["stride"][0]) for b in blocks])
    writer.add_array("stt.titanet.encoder.dilation", [int(b["dilation"][0]) for b in blocks])
    writer.add_array("stt.titanet.encoder.residual", [int(bool(b["residual"])) for b in blocks])
    writer.add_string("stt.titanet.pooling", str(dec.get("pool_mode", "attention")))

    # ----- tensors -----
    used: set[str] = set()

    def emit(src: str, dst: str):
        if src not in sd:
            raise KeyError(f"missing expected tensor: {src}")
        _add(writer, dst, _to_fp32(sd[src]))
        used.add(src)

    for b, block in enumerate(blocks):
        repeat = int(block["repeat"])
        for r in range(repeat):
            base = r * MCONV_STRIDE
            for src_suf, dst_suf in REPEAT_TABLE:
                src_suf = src_suf.format(i=base, j=base + 1, k=base + 2)
                emit(
                    f"encoder.encoder.{b}.mconv.{src_suf}",
                    f"enc.blocks.{b}.rep.{r}.{dst_suf}",
                )
        # Squeeze-excite lands after the last repeat's conv group, which is
        # two short of where the next group would have started.
        se = repeat * MCONV_STRIDE - 2
        for src_suf, dst_suf in SE_TABLE:
            emit(f"encoder.encoder.{b}.mconv.{se}.{src_suf}", f"enc.blocks.{b}.{dst_suf}")
        if block["residual"]:
            for src_suf, dst_suf in RES_TABLE:
                emit(f"encoder.encoder.{b}.{src_suf}", f"enc.blocks.{b}.{dst_suf}")

    for src, dst in HEAD_TABLE:
        emit(src, dst)

    # ----- unused-key audit -----
    unexpected = []
    for k in sd_keys - used:
        if k.startswith(EXPECTED_UNUSED_PREFIXES) or k.endswith(EXPECTED_UNUSED_SUFFIXES):
            continue
        unexpected.append(k)
    if unexpected:
        raise ValueError(f"{len(unexpected)} unmapped state_dict tensors, e.g. {sorted(unexpected)[:8]}")
    print(f"Emitted {len(used)} tensors ({len(sd_keys) - len(used)} skipped: frontend + classifier + BN counters)")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Wrote GGUF: {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("model", help="HF repo id or path to a .nemo checkpoint")
    ap.add_argument("--repo-id", default=None, help="HF repo id (for slug + provenance)")
    ap.add_argument("--out", default=None, help="Override output GGUF path")
    args = ap.parse_args()

    repo_id = args.repo_id or (
        args.model if "/" in args.model and not Path(args.model).exists() else None
    )
    if args.out:
        out_path = Path(args.out)
    else:
        if not repo_id:
            raise SystemExit("error: pass --repo-id (or a HF repo id) so the output slug can be derived")
        slug = slug_from_repo_id(repo_id)
        out_path = Path("models") / slug / gguf_name(slug, REFERENCE_DTYPE_LABEL)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    convert(args.model, out_path, repo_id=repo_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
