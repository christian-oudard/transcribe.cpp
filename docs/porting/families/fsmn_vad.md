# fsmn_vad

## Identity

- Family: `fsmn-vad`, architecture pattern `encoder-classifier`
- Variant: `fsmn-vad` (`funasr/fsmn-vad`, FunASR's
  `speech_fsmn_vad_zh-cn-16k-common`)
- Reference: FunASR 1.3.1, the only published implementation
- Reference dtype: F32
- 26 tensors, 1.7 MB at F32, 0.75 MB at Q8_0

Answers one question per frame: is anybody speaking. Not who, and not what.

It is here because diarization needs it. Clustering speaker embeddings counts
an extra speaker on most meetings, and the extra cluster is audible non-speech
that an energy threshold cannot tell from a voice. Five gates that are not
models were measured before this port and none worked; see
[`docs/diarization.md`](../../diarization.md).

## Forward pass

    fbank 80 mels, 25 ms / 10 ms, hamming
      -> LFR stack m=5 n=1, then CMVN          => [T, 400]
      -> in_linear1   400 -> 140   affine
      -> in_linear2   140 -> 250   affine, ReLU
      -> 4 x FSMN:    250 -> 128   linear, no bias
                      memory: depthwise conv over the last 20 frames,
                              added back to its own input
                      128 -> 250   affine, ReLU
      -> out_linear1  250 -> 140   affine
      -> out_linear2  140 -> 248   affine, softmax
      -> speech = 1 - posterior[silence]

The frontend is the kaldi fbank with LFR stacking this tree already had for
sensevoice and funasr_nano, which is most of why this model was chosen over
silero; the rest of why is that FSMN is feed-forward where silero carries an
LSTM state between calls.

The working layout in the graph is one column per frame, which is what the
linear layers want, since `ggml_mul_mat` contracts the fastest axis. The
memory convolution is the exception and runs over time, so the tensor is
permuted around it and back.

## Chunking is exact

The memory is left-only (`lorder` 20, `rorder` 0), so a frame's answer depends
on the 19 frames before it and nothing after. A chunk that carries those 19
frames as context computes the same numbers the whole-file graph would, and
they are discarded after the run.

This is not an optimization. One graph over a two-hour recording asks a card
for eight gigabytes; the model ran on eighteen-minute meetings and died on a
109-minute workshop. Chunked at five minutes, the same recording takes 12
seconds and matches the reference to the digit it matched before.

## Commands

Convert:

    nix develop -c uv run --project scripts/envs/funasr_nano \
      scripts/convert-fsmn-vad.py <dir with model.pt, config.yaml, am.mvn>

Reference tensors:

    nix develop -c uv run --project scripts/envs/funasr_nano \
      scripts/dump_reference_fsmn_vad_funasr.py --model <dir> \
      --audio samples/jfk.wav \
      --out build/validate/fsmn_vad/fsmn-vad/jfk/ref

Run, which prints speech regions as speaker rows:

    transcribe-cli --diarize -m models/fsmn-vad/fsmn-vad-Q8_0.gguf audio.wav

## Numerical parity

On `samples/jfk.wav`, against FunASR:

| Quant | max abs on the speech probability | frames decided the same | speech |
| --- | --- | --- | --- |
| F32 | 0.0027 | 100.0% | 80.0% vs 79.9% |
| Q8_0 | 0.0081 | 99.9% | — |

The per-frame agreement is the number that matters. A probability that differs
by 0.008 changes nothing downstream unless it crosses the threshold, and on
this clip that happens to one frame in a thousand.

Frame counts differ by two at the end of the clip (1096 against 1098): an LFR
edge convention, not a disagreement about the audio.

## Quant policy

Ship Q8_0. It halves an already small file and costs a tenth of a per cent of
the frame decisions.

The CMVN vectors must stay F32 and are named `frontend.cmvn.{shift,scale}` so
the quantizer's rule finds them. They are read back to host at load, and a
quantized vector is a smaller buffer than that read expects: the first attempt
at quantizing this model crashed inside `ggml_backend_tensor_get`.

## Capability Validation

| Capability | Mode | Command / test | Expected observable | Target | Status |
| --- | --- | --- | --- | --- | --- |
| Speech activity | offline | `transcribe-cli --diarize` with dumps | per-frame probability within 0.01 of the reference | MUST PASS | PASS |
| Frame decisions | offline | same | ≥99% of frames decided the same as the reference | MUST PASS | PASS (100% at F32) |
| Regions | offline | `--diarize` | contiguous speech rows, one anonymous speaker | MUST PASS | PASS |
| Long audio | offline | 109-minute recording | completes, memory flat in the length | MUST PASS | PASS (12 s) |
| Speaker attribution | — | — | none: every row is speaker 1 | OUT OF SCOPE — not a diarizer | N/A |
| Transcription | — | — | no text | OUT OF SCOPE | N/A |
| Streaming | — | — | — | OUT OF SCOPE — the offline path is exact and the streaming contract is unwritten | N/A |

The WER gate does not apply. What stands in its place is the frame agreement
above and, at the level anyone cares about, whether the regions improve
diarization: measured in `docs/diarization.md`, they fix the speaker count on
three of four AMI meetings and cost a fourth its sparsest speaker.

## Variant Notes

- `fsmn-vad`: the family baseline and the only published checkpoint. Trained
  on Chinese, and used here on English meetings, which is worth stating: speech
  detection is far less language-dependent than recognition, and the measured
  agreement with the reference says nothing about how well it detects English
  in particular. The AMI numbers are what say that.
