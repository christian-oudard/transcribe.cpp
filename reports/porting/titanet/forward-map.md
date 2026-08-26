# Forward map - titanet

Reference: NeMo `EncDecSpeakerLabelModel` @ pin in `scripts/envs/titanet`
Closest in-tree analog: src/arch/sortformer/ (encoder-only, non-text output)

TitaNet is an `encoder-embedding`. One run takes audio and returns a single
192-dim vector; there is no tokenizer, no decoder, no text and no timestamps.
Two clips of one voice land close together under cosine distance, which is the
piece the library is missing: sortformer tracks speakers but caps at four and
produces no vectors, so nothing can be clustered over a whole recording.

The whole graph is one shot over the clip. Nothing here is causal and nothing
streams: squeeze-excite at `se_context_size: -1` averages over the entire clip,
and the pooling takes a softmax across it, so every frame depends on every
other frame. A chunked evaluation is not an approximation of this, it is a
different function.

## Frontend

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Mel | `modules/audio_preprocessing.py` `AudioToMelSpectrogramPreprocessor` | [80, T] | `enc.mel.in` | `transcribe-mel.h`, n_fft 512, win 400, hop 160, hann, pre-emphasis 0.97 | sortformer/model.cpp |
| Normalize | same, `normalize: per_feature` | [80, T] | `enc.mel.in` | per-mel mean/std over the clip, not a fixed table | parakeet |

Dither is 1e-5 in training and must stay off here: it is noise, and it makes
the embedding non-deterministic for the same clip.

## Encoder

Five `JasperBlock`s, `filters [1024,1024,1024,1024,3072]`, `repeat
[1,3,3,3,1]`, `kernel [3,7,11,15,1]`, stride and dilation all 1, `residual
[0,1,1,1,0]`. Every conv is separable: depthwise then pointwise.

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Block repeat r | `parts/submodules/jasper.py` `JasperBlock.forward` | [C, T] | - | `ggml_conv_1d_dw` (k, same padding) then pointwise `ggml_mul_mat`, then BN, then ReLU | none in tree: no other family has a conv stack |
| BatchNorm | `nn.BatchNorm1d`, inference path | [C, T] | - | fold at load: `scale = g / sqrt(var + 1e-5)`, `shift = b - mean * scale`, then `ggml_mul` + `ggml_add` per channel | - |
| Squeeze-excite | `parts/submodules/tdnn_attention.py` `MaskedSEModule` | [C, T] | - | mean over T (`ggml_mean`), down `C -> C/8`, ReLU, up `C/8 -> C`, sigmoid, broadcast multiply | - |
| Residual | `JasperBlock` res path, blocks 1-3 | [C, T] | - | pointwise + folded BN on the skip, added before the block's last ReLU | - |
| Block out | `ConvASREncoder.forward` | [1024 or 3072, T] | `enc.block.0.out` .. `enc.block.4.out` | - | - |
| Encoder out | same | [3072, T] | `enc.out` | - | - |

Squeeze-excite at `se_context_size: -1` is a mean over the whole clip, so it is
not a windowed operation and has no length parameter. Getting this wrong is
silent: the shapes are identical either way and only the numbers move.

## Decoder

There is no decoder. `SpeakerDecoder` is pooling plus a projection.

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Clip statistics | `tdnn_attention.py` `get_statistics_with_mask` | [3072] x2 | - | mean over T; std as `sqrt(clamp(sum(m*(x-mean)^2), 1e-10))` | - |
| Attention input | `AttentivePoolLayer.forward` | [9216, T] | - | concat x, mean broadcast, std broadcast | - |
| Attention layer | `TDNNModule` then Tanh then Conv1d | [3072, T] | - | conv k=1 `9216 -> 128`, ReLU, folded BN, tanh, conv k=1 `128 -> 3072` | - |
| Alpha | `F.softmax(attn, dim=2)` | [3072, T] | - | softmax **over time**, per channel | - |
| Pooled | `get_statistics_with_mask(x, alpha)` | [6144] | `pool.out` | weighted mean and weighted std, concatenated | - |
| Embedding | `SpeakerDecoder.emb_layers.0` | [192] | `emb.out` | folded BN over 6144, then Linear `6144 -> 192` | - |

The 16681-way angular softmax head (`decoder.final`) classifies the training
speakers and is not converted. Nothing downstream may read it.

## Generation / KV Path

None. One forward pass, no autoregression, no cache.

## Capabilities And Language Controls

| Capability | Reference behavior | C++ API behavior | Family-doc Capability Validation row |
|------------|--------------------|------------------|--------------------------------------|
| speaker_embedding | 192-dim vector per clip | new capability KV, `stt.capability.speaker_embedding` | TODO |
| transcription | none | run returns no text; the API must refuse rather than return empty | TODO |
| timestamps | none | `MaxTimestamps` = none | TODO |
| languages | English training data | `en`, advertised as a hint only | TODO |

## Deviations From Closest Analog

- Every ported family so far is a transformer over frames. This is a
  convolutional stack with BatchNorm, which no other family in the tree has:
  BN has to be folded at load, and there is no LayerNorm to reset scale, so
  drift accumulates across five blocks with nothing to catch it. Gate every
  block rather than first/middle/last.
- The output is not a sequence. Validation is cosine agreement with the
  reference vector per clip plus equal error rate over verification trials,
  not WER, and the WER gate in stage 7 does not apply.

## Variant Notes

- `speakerverification_en_titanet_large`: the family baseline; 25M parameters,
  F32 reference dtype, 88.7 MB GGUF.
