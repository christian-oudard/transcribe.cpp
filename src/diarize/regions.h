// diarize/regions.h - turn a per-frame speech probability into stretches of
// speech.
//
// The seam between a detector and everything downstream. A detector answers
// per frame; a diarizer, a transcriber and a person all want stretches, and
// the rules that turn one into the other are policy rather than model: how
// long a gap ends a turn, and how short a blip is not speech at all.
//
// Kept here rather than in the detector because it is the detector's output
// that is per-frame, not its business what counts as a pause.

#pragma once

#include "windows.h"  // Region

#include <cstdint>
#include <vector>

namespace transcribe::diarize {

struct RegionRules {
    // Milliseconds per frame of the probability curve.
    double frame_ms = 10.0;
    // Above this a frame is speech. Halfway is where the detector's own
    // decision sits; the smoothing below is where the real policy lives.
    float threshold = 0.5f;
    // A gap shorter than this does not end a region. A breath inside a
    // sentence is not silence, and splitting there produces two regions that
    // every consumer has to put back together.
    int64_t min_silence_ms = 300;
    // A region shorter than this is not one. A single frame over the
    // threshold is a detector twitching, not somebody talking.
    int64_t min_speech_ms = 100;
};

// Speech regions from `n` per-frame probabilities, in milliseconds, ordered
// and non-overlapping.
std::vector<Region> regions_from_frames(const float * speech, int32_t n, const RegionRules & rules);

}  // namespace transcribe::diarize
