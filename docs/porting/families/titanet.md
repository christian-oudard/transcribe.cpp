# titanet

## Identity

- Family: `titanet`, architecture pattern `encoder-embedding`
- Variant: `speakerverification_en_titanet_large`
  (`nvidia/speakerverification_en_titanet_large`, rev `0dc382f4`)
- Reference: NeMo `EncDecSpeakerLabelModel`, the only implementation upstream
  ships; the pin is shared with the sortformer and parakeet envs
- Reference dtype: F32 (the NeMo speaker-model state_dict is fp32)
- 25M parameters, 88.7 MB at F32

A speaker embedding model. One run takes audio and returns a single 192-dim
vector; two clips of one voice land close together under cosine distance.
There is no tokenizer, no decoder, no text and no timestamps, and the run
returns no transcript however it is asked.

It is here to be clustered over. Every other diarizer in this library fixes the
number of speakers in its architecture -- sortformer at four -- so a recording
with more voices than that cannot be told apart at all. Embeddings turn the
count from a constant into a property of the audio.

## Notes

Nothing in this model is causal or chunkable. Squeeze-excite runs at
`se_context_size: -1`, a mean over the entire clip, and the attentive pooling
takes a softmax across it, so every frame depends on every other one. A window
of the audio is a different function rather than an approximation of this one,
which rules out evaluating a prefix and comparing it to the same prefix of a
longer run.

Two things cost a week between them and are worth stating plainly.

**BatchNorm epsilon is not one number.** NeMo's jasper blocks build theirs with
`eps=1e-3` (`jasper.py`, `_get_conv_bn_layer`); the pooling and embedding
BatchNorms take PyTorch's `1e-5` default. Folding all of them at 1e-5 left
block 1 at cosine 0.54 against the reference and the encoder diverging from
there, with every shape intact.

**`get_seq_len` is the ceiling here, not the floor.** The shared mel frontend
masked the final frame of every clip, which is right for a family whose
`get_seq_len` is `floor(n/hop)` and wrong for one whose is the ceiling. One
frame in eleven hundred moved the whole embedding, because the pooling takes
its statistics across the frames.

## Two files, one checkpoint

`cstr/titanet-large-GGUF` was published before this port. It describes the
model in its own vocabulary (`titanet.emb_dim`, `titanet.channels`,
`titanet.block_repeats`), carries no frontend block at all -- it bakes the mel
filterbank and the window in as tensors instead -- and stores pointwise
convolutions as plain matrices, `[in, out]` where the converter here writes
`[1, in, out]`.

None of that is information this tree needs and cannot get. Every stride and
dilation in TitaNet is 1, the activation is ReLU, the pooling is attentive
statistics, and the frontend is NeMo's standard 80-mel preprocessor; which
blocks carry a residual is read off the tensors rather than assumed, since a
block with a skip path has the convolution for it. The shape check ignores
axes of extent one, and the graph reshapes what it is handed.

What the published file has that this tree did not is both BatchNorm
epsilons, stated rather than hard-coded.

## Commands

Convert (needs the NeMo reference env):

    nix develop -c uv run --project scripts/envs/titanet \
      scripts/convert-titanet.py nvidia/speakerverification_en_titanet_large \
      --repo-id nvidia/speakerverification_en_titanet_large

Reference tensors, including per-sub-layer dumps inside one block for
bisecting a divergence:

    nix develop -c uv run --project scripts/envs/titanet \
      scripts/dump_reference_titanet_nemo.py encoder \
      --model nvidia/speakerverification_en_titanet_large \
      --audio samples/jfk.wav --sub-block 1 \
      --out build/validate/titanet/speakerverification_en_titanet_large/jfk/encoder/ref

Embed a clip (dumps carry the parity tensors):

    TRANSCRIBE_DUMP_DIR=... transcribe-cli -m <gguf> samples/jfk.wav

Diarize a recording:

    transcribe-cli --diarize [--speakers N] -m <gguf> meeting.wav

## Numerical parity

Against the oracle on `samples/jfk.wav`, F32:

| Tensor | max abs | rms | cosine |
| --- | --- | --- | --- |
| `enc.mel.in` | 1.9e-4 | — | — |
| `enc.out` | 5.4e-4 | 1.3e-5 | — |
| `pool.out` | 1.9e-5 | 1.3e-6 | 1.0000002 |
| `emb.out` | 1.1e-4 | 3.5e-5 | 0.99999845 |

The behavioural artifact is the cosine between two speakers' embeddings, which
is the only number the model is ever used through: reference 0.139264, ours
0.139642.

## Quant policy

| Quant | Size | cos vs reference (jfk / dots) | cos(jfk, dots) | AMI ES2011a confusion |
| --- | --- | --- | --- | --- |
| F32 | 88.7 MB | 1.000 / 1.000 | 0.1396 | 11.3% |
| Q8_0 | 41.3 MB | 0.99900 / 0.99926 | 0.1461 | 11.1% |
| Q5_K_M | 40.7 MB | 0.99901 / 0.99923 | 0.1437 | 11.1% |

Ship Q8_0. Q5_K_M saves 0.6 MB and buys nothing: most of this model is
depthwise convolutions and BatchNorm vectors, which are copied rather than
requantized at either preset (38 tensors requantized, 67 copied), so the
k-quant reaches almost the same tensors as the round one and only lowers their
precision.

The quantization is invisible where it matters. Diarization of an AMI meeting
gives the same speaker count and the same confusion to a tenth of a point at
either preset, which is the measurement that decides this rather than the
per-tensor cosines.

## Capability Validation

| Capability | Mode | Command / test | Expected observable | Target | Status |
| --- | --- | --- | --- | --- | --- |
| Speaker embedding | offline | `transcribe-cli -m <gguf> samples/jfk.wav` with dumps | `emb.out` 192 floats, cosine ≥ 0.999 against the reference vector | MUST PASS | PASS |
| Verification pair | offline | `dump_reference_titanet_nemo.py verify` against the same two clips | cosine within 1e-3 of the reference | MUST PASS | PASS (3.8e-4) |
| Diarization | offline | `transcribe-cli --diarize` on an AMI dev meeting | speaker rows, count from the eigengap | MUST PASS | PASS (4 of 4 on ES2011a; one over on three other meetings) |
| Speaker count as input | offline | `--speakers N` | exactly N speakers in the rows | MUST PASS | PASS |
| Transcription | — | any run | no text, ever | OUT OF SCOPE — not a transcription model | N/A |
| Timestamps | — | `MaxTimestamps` | none | OUT OF SCOPE — no text to stamp | N/A |
| Overlapping speech | offline | — | one speaker per window | OUT OF SCOPE — needs separation, not clustering | N/A |
| Streaming | — | — | — | OUT OF SCOPE — the graph is whole-clip by construction | N/A |
| Language hint | — | — | ignored | OUT OF SCOPE — a voice is not a language | N/A |

The WER gate does not apply: there is no transcript. What stands in its place
is the verification cosine above and the diarization confusion in
[`docs/diarization.md`](../../diarization.md), which also records what has been
measured and ruled out.
