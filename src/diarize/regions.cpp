// diarize/regions.cpp

#include "regions.h"

#include <cmath>

namespace transcribe::diarize {

std::vector<Region> regions_from_frames(const float * speech, int32_t n, const RegionRules & rules) {
    std::vector<Region> out;
    if (speech == nullptr || n <= 0 || rules.frame_ms <= 0.0) {
        return out;
    }
    const int64_t min_silence = static_cast<int64_t>(rules.min_silence_ms / rules.frame_ms);
    const int64_t min_speech  = static_cast<int64_t>(rules.min_speech_ms / rules.frame_ms);

    const auto ms = [&](int64_t frame) { return static_cast<int64_t>(std::llround(frame * rules.frame_ms)); };

    int64_t start = -1;  // first frame of the region being built
    int64_t quiet = 0;   // consecutive frames below the threshold since then
    for (int32_t i = 0; i <= n; ++i) {
        const bool voiced = (i < n) && speech[i] > rules.threshold;
        if (voiced) {
            if (start < 0) {
                start = i;
            }
            quiet = 0;
            continue;
        }
        if (start < 0) {
            continue;
        }
        ++quiet;
        // The end of the audio closes whatever is open, however short the
        // silence: there is no more of it to wait for.
        if (quiet < min_silence && i < n) {
            continue;
        }
        const int64_t end = i - quiet + 1;
        if (end - start >= min_speech) {
            out.push_back({ ms(start), ms(end) });
        }
        start = -1;
        quiet = 0;
    }
    return out;
}

}  // namespace transcribe::diarize
