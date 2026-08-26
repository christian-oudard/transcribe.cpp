# FSMN Voice Activity Detection

FunASR's [`funasr/fsmn-vad`](https://huggingface.co/funasr/fsmn-vad)
(`speech_fsmn_vad_zh-cn-16k-common`) ported to transcribe.cpp. Four FSMN
layers between two pairs of affine transforms, over kaldi fbank features with
LFR stacking: 26 tensors and 1.7 MB.

## Which file

Either of the two GGUFs of this checkpoint that exist:

- [`FunAudioLLM/fsmn-vad-GGUF`](https://huggingface.co/FunAudioLLM/fsmn-vad-GGUF)
  -- published, F32, 1.7 MB. Nothing has to be converted to use this model.
- what `scripts/convert-fsmn-vad.py` writes, which adds Q8_0 and states the
  frontend and the silence pdf in its metadata.

The weights are bit-identical; they differ in tensor names, hparam key
prefixes, and which axis of the memory kernel is time. The loader reads both
and the two produce the same rows on `samples/jfk.wav`, to the millisecond.

## What it's for

Answering one question per frame: is anybody speaking. Not who, and not what.

A run produces no text and no speaker identities. It takes a 16 kHz mono WAV
and emits speech regions on the speaker-segment surface (`t0_ms`, `t1_ms`,
`speaker_id`), every one of them speaker 1 — somebody was talking, and this
model does not know who.

It is here because the clustering diarizer needs it. Loudness cannot tell a
voice from a chair or a keyboard: on two AMI meetings an energy threshold
passed 97 and 172 seconds of audible non-speech, which then clustered
coherently and arrived as an extra speaker in a room of four. Handing these
regions to the diarizer
(`transcribe_titanet_diarize_ext::speech_ms`, `transcribe-cli --speech`) fixes
the count on three of four meetings; the measurements are in
[docs/diarization.md](../diarization.md).

The memory is left-only, so a frame's answer never depends on audio after it.
That makes chunking exact rather than approximate — a two-hour recording is
processed five minutes at a time with nineteen frames of context and produces
the numbers a whole-file graph would.

Trained on Chinese and used here on English meetings. Speech detection is far
less language-dependent than recognition, but the parity numbers below say
nothing about English in particular: they say this port agrees with FunASR.

Licensed under the terms on the upstream model card.

## Download

| Quantization | Size | Frames decided as the reference does |
| --- | ---: | ---: |
| F32  | 1.7 MB  | 100.0% |
| Q8_0 | 0.75 MB |  99.9% |

Ship Q8_0. Its per-frame probability differs from the reference by at most
0.008, which changes a decision only when it crosses the threshold, and on the
reference clip that happens to one frame in a thousand.

## Notes

The normalization vectors (`frontend.cmvn.{shift,scale}`) must stay F32. They
are read back to host at load, and a quantized vector is a smaller buffer than
that read expects; the quantizer's policy keeps them by suffix.

See [docs/porting/families/fsmn_vad.md](../porting/families/fsmn_vad.md) for
the forward pass, the parity table and the capability validation.
