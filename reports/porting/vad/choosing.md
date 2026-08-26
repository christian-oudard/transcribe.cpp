# Choosing a voice activity detector

Diarization needs one, and this records which to port and why, so the decision
does not have to be made twice.

## Why a model rather than a threshold

The clustering counts an extra speaker on three of four AMI meetings, and the
extra cluster is not a speaker: on IS1008a it is 97.5 seconds that the
reference says is 99% not speech, on ES2011c 172.5 seconds at 92%. Audible
non-speech -- a chair, a keyboard, paper -- clusters coherently and arrives at
the counting stage looking exactly like a quiet participant.

Five ways of removing it have been measured and none works:

| gate | counts (reference 4 each) | cost |
|------|---------------------------|------|
| energy, as shipped | 4, 5, 5, 5 | none |
| energy, 2x louder | 4, 5, 5, 5 | none |
| energy, 4x louder | 3, 5, 5, 4 | 17.7% of speech on one meeting |
| syllabic modulation | 3, 4, 4, 4 | 32-39% of speech on two meetings |
| sortformer's speech | 3, 4, 4, — | loses a speaker on ES2011a |
| transcriber's words | 4, 4, 5, — | under 3% missed |
| words and sortformer, unioned | 4, 4, 5, — | under 1% missed |

The word mask is the best of them and still leaves one count wrong, because on
that meeting the transcriber produced words over the non-speech too. A mask is
only as honest as whatever drew it, and none of these was drawn by something
trained to answer "is this speech".

## Which one

**FSMN-VAD** (`speech_fsmn_vad_zh-cn-16k-common`, FunASR), not silero.

The deciding factor is what this tree already has rather than which model is
better in the abstract; both are small and both are used in production
elsewhere.

- The frontend is already here. `transcribe-kaldi-fbank.h` is kaldi-style HTK
  mel fbank with the LFR stacking FunASR models take, written for sensevoice
  and funasr_nano. FSMN-VAD takes exactly that. silero brings its own STFT
  frontend, which would be new code with its own parity gate.
- The block is nearly here. sensevoice already implements an FSMN memory block
  as a depthwise conv over time (`attn_fsmn_w`, `[kernel, 1, d_model]`), which
  is the same operator the VAD is built from.
- The state is simpler. silero v5 carries an LSTM hidden state between calls,
  which means a streaming contract, a state to reset, and a new class of bug.
  FSMN is feed-forward with a fixed left context: the same audio gives the same
  answer whatever came before it, which is what an offline diarizer wants.
- Licensing is comparable and permissive for both.

## What the port has to produce

Not a transcript, and not speaker rows: a per-frame speech probability, which
`src/diarize/windows.cpp` turns into the regions the extension already accepts
(`transcribe_titanet_diarize_ext::speech_ms`). That seam exists and is tested,
so the VAD lands behind an interface with a measurement waiting for it: the
same four meetings, where the target is four speakers on all of them without
losing more than a couple of per cent of the speech.
