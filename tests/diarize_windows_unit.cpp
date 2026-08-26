// Unit test: windowing and row assembly, the two ends of the diarization
// pipeline that are pure arithmetic over timings.

#include "diarize/windows.h"

#include <cmath>
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

// Loud audio for `seconds`, then silence for `seconds`, alternating.
std::vector<float> alternating(int rate, int seconds, int blocks) {
    std::vector<float> pcm;
    for (int b = 0; b < blocks; ++b) {
        for (int i = 0; i < rate * seconds; ++i) {
            pcm.push_back((b % 2 == 0) ? std::sin(static_cast<float>(i) / 8.0f) : 0.0f);
        }
    }
    return pcm;
}

}  // namespace

int main() {
    WindowConfig cfg;
    const int    rate = cfg.sample_rate;

    // Silence is dropped, so a pause cannot cluster into a speaker of its own.
    {
        std::vector<float>  pcm     = alternating(rate, 6, 4);  // loud, quiet, loud, quiet
        std::vector<Window> windows = speech_windows(pcm.data(), static_cast<int32_t>(pcm.size()), cfg);
        check(!windows.empty(), "the loud stretches produce windows");
        // A window straddling the boundary has speech in it and is kept; what
        // must not survive is one that lies wholly inside a silent block.
        bool wholly_silent = false;
        for (const Window & w : windows) {
            const double from = static_cast<double>(w.from) / rate;
            const double to   = static_cast<double>(w.to) / rate;
            if (static_cast<int>(from / 6.0) % 2 == 1 && static_cast<int>(from / 6.0) == static_cast<int>(to / 6.0)) {
                wholly_silent = true;
            }
        }
        check(!wholly_silent, "no window lies wholly inside the silence");
    }

    // A recording shorter than one window is still one window: a short answer
    // is a speaker turn.
    {
        std::vector<float> pcm(static_cast<size_t>(rate), 0.5f);
        std::vector<Window> windows = speech_windows(pcm.data(), static_cast<int32_t>(pcm.size()), cfg);
        check(windows.size() == 1 && windows[0].from == 0, "a short clip is one window");
    }

    // Consecutive windows by one speaker merge into a row that spans them.
    {
        std::vector<Window>  w      = { { 0, 48000 }, { 24000, 72000 }, { 48000, 96000 } };
        std::vector<int32_t> labels = { 0, 0, 0 };
        std::vector<Row>     rows   = rows_from_labels(w, labels, rate);
        check(rows.size() == 1, "one speaker across three windows is one row");
        check(rows[0].t0_ms == 0 && rows[0].t1_ms == 6000, "the row spans first start to last end");
    }

    // A change of speaker closes the row, and the rows meet rather than
    // overlap even though the windows do.
    {
        std::vector<Window>  w      = { { 0, 48000 }, { 24000, 72000 }, { 48000, 96000 } };
        std::vector<int32_t> labels = { 0, 1, 1 };
        std::vector<Row>     rows   = rows_from_labels(w, labels, rate);
        check(rows.size() == 2, "a change of speaker closes the row");
        check(rows[0].speaker == 0 && rows[1].speaker == 1, "the rows carry their speakers");
        check(rows[1].t0_ms == 1500 && rows[1].t1_ms == 6000, "the second row starts at its first window");
    }

    // A gap in the windows -- silence the windows were dropped from -- closes
    // the row even for one speaker.
    {
        std::vector<Window>  w      = { { 0, 48000 }, { 160000, 208000 } };
        std::vector<int32_t> labels = { 0, 0 };
        std::vector<Row>     rows   = rows_from_labels(w, labels, rate);
        check(rows.size() == 2, "silence between two turns by one person is two rows");
    }

    // Mismatched inputs produce nothing rather than reading past an end.
    {
        std::vector<Window>  w      = { { 0, 48000 } };
        std::vector<int32_t> labels = { 0, 1 };
        check(rows_from_labels(w, labels, rate).empty(), "a label per window, or no rows");
    }

    if (failures == 0) {
        std::printf("diarize_windows_unit: OK\n");
    }
    return failures == 0 ? 0 : 1;
}
