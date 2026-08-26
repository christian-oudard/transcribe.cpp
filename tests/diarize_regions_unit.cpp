// Unit test: per-frame speech probabilities into stretches of speech.
//
// This is policy rather than model -- what counts as a pause, and what is too
// short to be speech -- so it is the part that can be wrong without any tensor
// being wrong, and the part a detector's output is judged through.

#include "diarize/regions.h"

#include <cstdio>
#include <vector>

using namespace transcribe::diarize;

namespace {

int failures = 0;

void check(bool ok, const char * what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

// A curve at 10 ms per frame: pairs of (value, how many frames).
std::vector<float> curve(std::initializer_list<std::pair<float, int>> runs) {
    std::vector<float> out;
    for (const auto & [value, n] : runs) {
        out.insert(out.end(), static_cast<size_t>(n), value);
    }
    return out;
}

}  // namespace

int main() {
    RegionRules rules;  // 10 ms frames, 0.5 threshold, 300 ms silence, 100 ms speech

    // One stretch of speech is one region, and its edges are where the
    // probability crossed.
    {
        std::vector<float> c = curve({ { 0.1f, 50 }, { 0.9f, 100 }, { 0.1f, 50 } });
        auto               r = regions_from_frames(c.data(), static_cast<int32_t>(c.size()), rules);
        check(r.size() == 1, "one stretch is one region");
        if (r.size() == 1) {
            check(r[0].t0_ms == 500 && r[0].t1_ms == 1500, "the region spans the speech");
        }
    }

    // A breath inside a sentence is not the end of a turn.
    {
        std::vector<float> c = curve({ { 0.9f, 100 }, { 0.1f, 20 }, { 0.9f, 100 } });
        auto               r = regions_from_frames(c.data(), static_cast<int32_t>(c.size()), rules);
        check(r.size() == 1, "a 200 ms gap does not split a region");
    }

    // A pause between speakers is.
    {
        std::vector<float> c = curve({ { 0.9f, 100 }, { 0.1f, 60 }, { 0.9f, 100 } });
        auto               r = regions_from_frames(c.data(), static_cast<int32_t>(c.size()), rules);
        check(r.size() == 2, "a 600 ms gap splits a region");
    }

    // A single frame over the threshold is a detector twitching.
    {
        std::vector<float> c = curve({ { 0.1f, 50 }, { 0.9f, 3 }, { 0.1f, 50 } });
        check(regions_from_frames(c.data(), static_cast<int32_t>(c.size()), rules).empty(),
              "a 30 ms blip is not speech");
    }

    // Speech that runs to the end of the audio is closed by the end of the
    // audio, however short the trailing silence.
    {
        std::vector<float> c = curve({ { 0.1f, 50 }, { 0.9f, 100 } });
        auto               r = regions_from_frames(c.data(), static_cast<int32_t>(c.size()), rules);
        check(r.size() == 1, "the last region is closed at the end of the clip");
        if (r.size() == 1) {
            check(r[0].t1_ms == 1500, "and it runs to the last speaking frame");
        }
    }

    // Silence throughout is no regions, not one empty one.
    {
        std::vector<float> c = curve({ { 0.05f, 200 } });
        check(regions_from_frames(c.data(), static_cast<int32_t>(c.size()), rules).empty(), "silence is no regions");
    }

    // Nothing in, nothing out.
    check(regions_from_frames(nullptr, 0, rules).empty(), "no frames, no regions");

    if (failures == 0) {
        std::printf("diarize_regions_unit: OK\n");
    }
    return failures == 0 ? 0 : 1;
}
