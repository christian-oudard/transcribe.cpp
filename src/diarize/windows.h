// diarize/windows.h - cut a recording into the windows a speaker embedding is
// taken over, and turn the labels back into who-spoke-when rows.
//
// The two ends of the diarization pipeline, both of them pure arithmetic over
// timings. What happens in between is a neural network and a clustering.

#pragma once

#include <cstdint>
#include <vector>

namespace transcribe::diarize {

// One window of audio, in samples, half-open.
struct Window {
    int32_t from;
    int32_t to;
};

// One who-spoke-when row, in milliseconds.
struct Row {
    int64_t t0_ms;
    int64_t t1_ms;
    int32_t speaker;  // 0-based; the caller adds whatever offset it publishes
};

struct WindowConfig {
    int32_t sample_rate = 16000;
    // A speaker embedding needs enough voice to be a voice, and this is the
    // measurement rather than a convention. Over two known speakers, cosine
    // distance between windows of one voice against windows of two:
    //
    //   1.5 s windows   within median 0.45, p90 0.69, max 0.96
    //                   between median 0.93, p10 0.81, min 0.61
    //   3.0 s windows   within median 0.25, p90 0.39, max 0.69
    //                   between median 0.90, p10 0.78, min 0.55
    //
    // At 1.5 s the two distributions overlap and no threshold separates them;
    // at 3 s they do not, and every threshold from 0.55 to 0.80 recovers
    // exactly the two speakers. The cost is that a turn shorter than a window
    // is blurred into its neighbour, which is what a real segmenter in front
    // of this would fix.
    float window_seconds = 3.0f;
    float hop_seconds    = 1.5f;
    // Windows quieter than this fraction of the recording's loud parts are
    // dropped as silence. Deliberately crude: it is here so that a pause does
    // not cluster into a speaker of its own, not to be a voice activity
    // detector. A real segmenter belongs in front of this.
    float silence_floor = 0.05f;
};

// One stretch the caller says is speech, in milliseconds.
struct Region {
    int64_t t0_ms;
    int64_t t1_ms;
};

// The windows worth embedding: uniform windows over the recording, minus the
// ones with no speech in them.
//
// With regions, a window is kept when at least half of it falls inside one,
// and loudness is not consulted: whoever supplied the regions knows more about
// where the speech is than an energy threshold does. Without them, the crude
// gate above is all there is.
std::vector<Window> speech_windows(const float *              pcm,
                                   int32_t                    n_samples,
                                   const WindowConfig &       config,
                                   const std::vector<Region> & regions = {});

// Turn one label per window back into rows. Consecutive windows by the same
// speaker merge, and a gap in the windows -- silence, or the end of the
// recording -- closes the row.
//
// A row runs from the start of its first window to the end of its last, so
// overlapping windows produce rows that meet rather than overlap.
std::vector<Row> rows_from_labels(const std::vector<Window> &  windows,
                                  const std::vector<int32_t> & labels,
                                  int32_t                      sample_rate);

}  // namespace transcribe::diarize
