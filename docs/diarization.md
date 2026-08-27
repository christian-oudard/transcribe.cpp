# Diarization

Who spoke when, for a recording with more people in it than any end-to-end
diarizer here can hold.

## The pipeline

    audio -> windows -> embeddings -> affinity -> spectrum -> labels -> rows

- `src/diarize/windows.cpp` cuts the recording into 3-second windows at a
  1.5-second hop and drops the quiet ones. This is not voice activity
  detection; it exists so that a pause does not cluster into a speaker of its
  own.
- `src/arch/titanet/diarize.cpp` embeds each window: one 192-dimensional
  vector per window, one graph built and computed per window, so memory stays
  flat however long the recording is.
- `src/diarize/count.cpp` decides how many speakers there are and which window
  belongs to which, by the normalized maximum eigengap (NME-SC, Park et al.
  2019) over a few hundred sampled windows, then k-means over the leading
  eigenvectors.
- `src/diarize/cluster.cpp` attributes every remaining window to the nearest
  cluster mean, and holds the older agglomerative path for a caller who
  supplies a distance instead of a count.

A caller who knows the number of speakers should say so
(`transcribe_titanet_diarize_ext::num_speakers`, `transcribe-cli --speakers`).
It is a fact about the room, and it outranks anything inferred from distances
between voices.

A caller who knows where the speech is should say that too
(`::speech_ms`, `transcribe-cli --speech`), and it is worth more. Given
regions, a window is embedded when at least half of it falls inside one and
loudness is not consulted at all. Measured with regions taken from the word
timings of a transcriber that had already run over the same audio:

| meeting | found | confusion | with regions |
|---------|-------|-----------|--------------|
| ES2011a | 4 → 4 | 11.3% → 11.1% | 2.6% missed |
| IS1008a | 5 → **4** |  4.6% → **4.1%** | 0.2% missed |
| ES2011c | 5 → 5 |  8.4% → **7.3%** | 1.3% missed |

Better on every meeting, and one of the two wrong counts is fixed. The one
that is not is the meeting where the transcriber produced words over the
non-speech as well, so the mask kept it: a mask is only as honest as whatever
drew it.

## Where it stands

Speaker confusion against the AMI dev references (BUT diarization setup),
frame-level at 10 ms over reference speech, with the best one-to-one mapping
between hypothesis and reference speakers. Overlap is excluded by design.

| meeting | reference | found | confusion |
|---------|-----------|-------|-----------|
| ES2011a | 4         | 4     | 11.3%     |
| IS1008a | 4         | 5     |  4.6%     |
| ES2011c | 4         | 5     |  8.4%     |
| TS3004a | 4         | 5     | 14.2%     |

Average 9.6%, against 8.4% published for x-vector agglomerative clustering and
6.3% for VBx on this corpus with the clustering as the only variable. Missed
speech is under half a per cent, so the windowing covers the audio and every
error is attribution.

Cost: about 18 seconds for 18.6 minutes of audio on a laptop RTX 4070, which
is 60x realtime, nearly all of it in the per-window embedding.

## With a voice activity detector in front

`models/fsmn-vad` answers per frame whether anybody is speaking, and its
regions can be handed to the diarizer (`--speech`, or
`transcribe_titanet_diarize_ext::speech_ms`). This is what the extra cluster
needed. Measured on the four meetings, at three widths -- as the detector
draws them, and dilated on each side, since a detector's edges are tight to
the speech where an embedding window wants context:

| regions | ES2011a | ES2011c | IS1008a | TS3004a |
|---------|---------|---------|---------|---------|
| none (energy) | 4, 11.0% | 5, 8.2% | 5, 4.6% | 5, 13.4% |
| as detected | 3, **9.6%** | **4**, **7.5%** | 5, 4.5% | **4**, **12.6%** |
| +100 ms | 3, 9.8% | **4**, 7.5% | 5, **4.4%** | 5, 13.5% |
| +200 ms | 4, 11.7% | **4**, 7.7% | 5, 4.6% | 5, 13.6% |

Pass the regions as the detector draws them. That is best or equal on
confusion for all four meetings and fixes the count on two of the three that
were over; dilating gives the improvement back a meeting at a time, and at
200 ms the non-speech is readmitted and everything is roughly as it was.

A region shorter than half a second is worth dropping before the windows are
drawn. The detector answers whether a frame is speech and a cough, a door or a
laugh gets a yes; nobody says a word in under half a second, so what is left
after that floor is people talking. It takes the last over-count out --
IS1008a goes to four -- and improves confusion on three of the four:

| regions | ES2011a | ES2011c | IS1008a | TS3004a |
|---------|---------|---------|---------|---------|
| as detected | 3, 9.6% | 4, 7.5% | 5, 4.5% | 4, 12.6% |
| at least 500 ms | 3, **9.5%** | 4, **7.4%** | **4**, 4.5% | 4, **12.4%** |
| at least 1 s | 3, 9.5% | 4, 7.6% | **4**, 4.5% | 4, **12.2%** |

Both thresholds fix the count, which is what says this is a floor rather than
a constant fitted to four recordings.

ES2011a comes back with three, and three is the honest answer: it has a
speaker who talks for 19.5 seconds in 13.6 minutes, and this pipeline has
never resolved them, with the detector or without it. What it used to report
as that meeting's fourth speaker was a cluster that is 90% non-speech. The
detector removes the cluster; it does not remove the speaker, whose audio it
keeps 94.9% of.

An earlier version of this section reported that width as costing ES2011a its
count *and* doubling its confusion, to 29.1%. That number was a scoring error,
not a result. The scorer enumerated mappings by permuting the hypothesis
clusters against the reference speakers and zipping them, and zip truncates:
with three clusters for four speakers the fourth speaker could not be mapped
at all, so every frame of the largest well-recognised speaker counted as
confused. It is 9.6%, and the detector improves every meeting rather than
trading one against three.

## Known limits

**The count runs one over, without the detector in front.** Three of the four
meetings report five speakers where there are four, and the extra cluster is
not a stray -- it holds between 9 and 15 per cent of the talk time. With the
detector's regions two of those three come back at four; the section above has
the numbers. What follows is why the extra one appears at all, which is worth
keeping: it is the argument for the detector.

It is also not a speaker. Scoring each hypothesis cluster against the reference
by what it actually contains, on IS1008a:

| cluster | length | what the reference says it is |
|---------|--------|-------------------------------|
| S3 | 487.5 s | MIO086 79% |
| S2 | 202.5 s | FIE073 74% |
| S5 | 166.5 s | MIE085 76% |
| S4 |  93.0 s | FIE038 83% |
| S1 |  97.5 s | **not speech, 99%** |

Four clusters are the four speakers. The fifth is a minute and a half of
audible non-speech that the energy gate let through and the clustering then
grouped, correctly, as a thing that is not any of the four voices. The count is
right about what it was given; the windowing gave it something that is not a
person.

**Short recordings were counted by how badly the graph broke.** The sweep over
pruning widths starts at two neighbours per window. On a long meeting that is
never chosen -- there are hundreds of windows and a width that sparse loses to
wider ones -- but on thirty seconds of two voices the whole sweep is p in
{2, 3, 4}, and at 2 and 3 the affinity graph falls into pieces:

| p | eigenvalues | pieces | k | gap/p |
|---|-------------|--------|---|-------|
| 2 | -0.000 0.000 0.055 0.315 | 2 | 8 | 0.189 |
| 3 | -0.000 -0.000 0.217 0.366 | 2 | 6 | 0.098 |
| 4 | -0.000 0.019 0.371 0.618 | 1 | 2 | 0.088 |

The multiplicity of eigenvalue zero is the number of connected components, so
on a broken graph the widest eigengap measures the break rather than the room,
and the NME criterion -- widest gap for the width -- then prefers the most
broken candidate. Thirty seconds of two voices came back as eight speakers.

Widths that disconnect the graph are now passed over when any width does not,
which picks p=4 above and answers two. The four meetings are unchanged to the
digit, which is the point: on those, every candidate width is already
connected and the rule never binds.

What it does not fix is the count on a chain-like geometry. Two synthetic
groups of slowly rotating vectors -- a voice drifting rather than sitting in a
ball -- estimate at 4, 8, or 12 speakers with the rule and without it. No real
recording here shows that shape, so it stays a note rather than a change, but
it is the first place to look at the count again.

**A speaker with twenty seconds in fourteen minutes is not resolved.** ES2011a
has one, and neither the energy gate nor the detector's regions changes that:
their windows are too few to be a cluster, and the count estimator does not
see them as one. It is not a gating problem -- 94.9% of their speech is inside
the detector's regions -- and no width of window or region tried here recovers
them. What would is a different question from the one on this page: not how
many people are talking, but whether a handful of windows scattered across a
meeting belong together.

**Overlapping speech is out of scope.** A window is attributed to one speaker.
In the published AMI numbers overlap accounts for about twenty points of
diarization error against one or two for the choice of clustering, so this is
the largest single thing not done -- and it is a different problem, needing
separation rather than clustering.

## Tried and ruled out

Both of these are measured on the four meetings above. Neither is worth
redoing without new evidence.

**Merging clusters that are close.** The natural repair for the count: merge
two groups when their centres are nearer to each other than their own members
typically are. The ratio of separation to spread for the closest pair came out
as 1.40, 3.06, 2.22 and 0.99 -- backwards. The meetings whose count is wrong
have the *most* separated groups, and the only pair close enough to merge was
two different people; merging it fixed that meeting's count and took its
confusion from 14.2% to 21.7%.

**Temporal smoothing.** A Viterbi pass over the window sequence with a
switching cost estimated from the data made three meetings worse, 9.6% to
11.8% on average. A turn in these meetings is two or three windows long, so
the window rate and the turn rate are the same order: there is no held floor
for a transition prior to exploit, and what it gains on an ambiguous window it
loses by running past a real change of speaker. It would pay on a lecture,
which is a different shape of recording. This also weakens the case for VBx
here, since its advantage over plain clustering is the same prior.

**Longer windows.** A three-second embedding still carries a lot of what the
speaker happened to be saying, so averaging more speech into each vector should
tighten the clusters. It does not, and it costs the time resolution:

| window | ES2011a | IS1008a | ES2011c | TS3004a | count |
|--------|---------|---------|---------|---------|-------|
| 3.0 s  | 11.1%   | 4.6%    | 8.3%    | 14.1%   | 4/5/5/5 |
| 4.5 s  | 13.5%   | 5.8%    | 10.1%   | 16.2%   | 4/5/5/5 |
| 6.0 s  | 13.0%   | 7.1%    | 13.6%   | 18.3%   | 4/5/5/5 |

Worse on every meeting, and the speaker count does not move at any length.
Whatever splits one person into two survives averaging six seconds of them
together.

**More windows in the spectrum.** The count is read from an affinity matrix
over 400 sampled windows; sampling 800 changes no count and no confusion by
more than a tenth of a point, and takes five to ten times as long -- 105 s
against 18 s on one meeting, 246 s on another -- because the eigendecomposition
is cubic in the sample. 400 is not a compromise, it is past the point where
more stops helping.

**Unioning the masks.** Words and sortformer regions together, since each
finds speech the other misses: 4, 5 and 4 speakers on the three meetings
measured, against 4, 5 and 4 for words alone, with confusion a few tenths
worse. A more permissive mask does not help when the problem is a mask that is
already too permissive.

**Turn-bounded windows.** Placing windows inside a segmenter's turns rather
than on a fixed grid, so that none straddles a speaker change. Scored on
windows a single speaker fully covers, it was no better than the uniform grid
at 3 seconds (77.4% against 80.2% purity), and only helps at 1.5 seconds,
where the uniform grid is bad for exactly this reason. Not worth a second
model in the pipeline.

**A louder silence gate.** If the fifth cluster is non-speech that got through,
raise the floor. Measured at 0.10 and 0.20 of the 95th-percentile window level
against the 0.05 shipped: at 0.10 no count changes at all, and at 0.20 one
meeting loses a real speaker and 17.7% of its speech. The non-speech is not
quiet -- it survives a fourfold louder threshold -- so it is audible: noise,
laughter, typing, or a chair. Loudness cannot separate it from a voice.

**A different embedding model.** This was the standing conclusion here until it
was tested. Same windows, same clustering, same counting, ECAPA-TDNN instead of
TitaNet -- a different architecture trained on different data -- and it returns
the same five speakers on IS1008a. Two independent embeddings making the same
mistake is not an embedding problem.

**A syllabic-rate gate.** People produce syllables two to eight times a second,
so a window of speech should have most of its envelope movement in that band
and a window of noise should not. It does separate them in distribution -- on
IS1008a, speech at 0.37 [0.30-0.43] against non-speech at 0.25 [0.17-0.37] --
and the counts improve: three of the four meetings come back with four
speakers instead of five.

It is still not usable, because the threshold that removes the non-speech
removes a third of the speech with it. Implemented and measured end to end:

| meeting | found | missed speech | confusion |
|---------|-------|---------------|-----------|
| ES2011a | 3     | 38.8%         | 19.4%     |
| IS1008a | 4     |  4.2%         |  3.2%     |
| ES2011c | 4     | 32.2%         |  3.7%     |
| TS3004a | 5     |  7.7%         | 13.5%     |

Two meetings lose a third of their speech to buy a speaker count. That is the
wrong trade, and it is worth noticing that the experiment which chose the
threshold measured only the count -- the coverage cost was visible in the
window totals at the time and went unweighed.

**Merging labels that trade places.** People do not swap the floor with no gap
between them, so two labels that alternate within a second, over and over,
should be one voice changing rather than two people talking. It is how the
workshop's over-split was diagnosed: its three biggest labels traded places
684 times in 109 minutes, which no two people do.

As a repair it does not work, because a dominant speaker alternates with
everybody constantly and looks exactly the same. Measured as a share of the
smaller cluster's turns, against the AMI meetings where every cluster is
known to be a different speaker:

| pair | share | truth |
|------|-------|-------|
| IS1008a S02/S04 | 1.50 | two people |
| IS1008a S02/S03 | 1.43 | two people |
| ES2011a S01/S03 | 1.06 | two people |
| TS3004a S02/S03 | 0.97 | two people |
| workshop S01/S03 | 0.94 | **one voice** |
| ES2011c S01/S02 | 0.89 | two people |

The one known split sits in the middle of the distribution of genuine pairs,
and several real pairs score higher. The signal is real when the answer is
already known from somewhere else -- a presenter holding 68% of the floor
cannot be three clusters of 25% -- and worthless as an automatic test.

## What would move it

Speech detection, which this deliberately does not do. Windows quieter than a
fraction of the recording's loud parts are dropped and nothing more, and the
measurement above says exactly what that costs: a minute and a half of audible
non-speech per meeting, coherent enough to cluster, arriving at the counting
stage as a fifth speaker.

A trained voice activity detector, which is a model rather than a feature. Two
substitutes have now been measured and neither works: a hand-rolled syllabic
gate removes a third of the speech at the threshold that removes the noise, and
sortformer's own speech regions -- it decides on more than loudness, and does
fix the count on the two meetings where the extra cluster is noise -- lose a
speaker on a third meeting, because a diarizer capped at four speakers reports
speech only where it has assigned one.

silero-vad is about a megabyte and does exactly this job. That is the shape of
the remaining work: a small port, not a threshold.

One thing no gate will fix: ES2011a has a speaker who talks for 25 seconds in
14 minutes. Every filter tried drops them, and the count then says three. A
speaker that brief is at the edge of what clustering can find at all.

Note that this is not the same as turn-bounded windows, which were tried and
did not help: that experiment moved where the windows *start*, scored on
windows a single speaker fully covers, and it could not see this because it had
already thrown the non-speech windows away.
