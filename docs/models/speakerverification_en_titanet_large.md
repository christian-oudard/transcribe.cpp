# TitaNet Large

NVIDIA's [`nvidia/speakerverification_en_titanet_large`](https://huggingface.co/nvidia/speakerverification_en_titanet_large)
ported to transcribe.cpp. Five ContextNet blocks with squeeze-excite,
attentive statistics pooling, and a projection to a 192-dimensional vector.

## Which file

Either of the two GGUFs of this checkpoint that exist:

- [`cstr/titanet-large-GGUF`](https://huggingface.co/cstr/titanet-large-GGUF)
  -- published, F16, 44.6 MB. Nothing has to be converted to use this model.
- what `scripts/convert-titanet.py` writes, which adds F32 and Q5_K_M.

Their weights agree with the converter's to 4.6e-4 relative, which is F16
rounding and nothing else, over all 105 tensors. They differ in tensor names,
in hparam vocabulary -- `titanet.channels` and `titanet.block_kernels` against
`stt.titanet.encoder.filters` and `.kernel` -- and in whether a pointwise
convolution is stored as the matrix it is. The loader reads both, and both
produce identical speaker rows.

The published file states the two BatchNorm epsilons, which this tree had
hard-coded. The encoder's is 1e-3 rather than PyTorch's 1e-5 default, and
getting it wrong is not subtle: block 1 came back at cosine 0.54 against the
reference.

## What it's for

Turning a clip into one vector in which two recordings of the same voice land
close together under cosine distance. There is no tokenizer, no decoder and no
text: a run produces an embedding, and with `transcribe_run_params::diarize`
on, the recording is cut into windows, each is embedded, and windows whose
vectors point the same way become one speaker.

That is what lifts the speaker ceiling. Every end-to-end diarizer here fixes
the number of speakers in its architecture — sortformer at four — so a
recording with more voices than that cannot be told apart at all. Clustering
has no such number in it, and the count comes from the data: the largest gap
in the eigenvalues of the affinity graph (NME-SC), or from the caller when
they know it (`--speakers`).

Nothing in the graph is causal or chunkable. Squeeze-excite averages over the
whole clip and the pooling takes a softmax across it, so every frame depends
on every other; a window of the audio is a different function rather than an
approximation of this one.

Licensed CC-BY-4.0. Ported from revision `0dc382f4`.

## Download

| Quantization | Size | Cosine against the reference embedding |
| --- | ---: | ---: |
| F32  | 88.7 MB | 0.99999845 |
| Q8_0 | 41.3 MB | 0.99900 |

Ship Q8_0. Q5_K_M saves another 0.6 MB and buys nothing: most of this model is
depthwise convolutions and BatchNorm vectors, which are copied rather than
requantized at either preset.

## Diarization quality

Speaker confusion against the AMI dev references, frame-level with the best
one-to-one speaker mapping, using an energy gate for speech:

| meeting | reference | found | confusion |
| --- | ---: | ---: | ---: |
| ES2011a | 4 | 4 | 11.3% |
| IS1008a | 4 | 5 |  4.6% |
| ES2011c | 4 | 5 |  8.4% |
| TS3004a | 4 | 5 | 14.2% |

Against 8.4% published for x-vector agglomerative clustering and 6.3% for VBx
on this corpus. The counts that run one over are audible non-speech clustering
as a fifth voice, which a voice activity detector in front fixes; see
[docs/diarization.md](../diarization.md), which also records seven repairs
that were measured and did not work.

See [docs/porting/families/titanet.md](../porting/families/titanet.md) for the
forward pass and the parity table.
