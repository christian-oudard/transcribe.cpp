// diarize/windows.cpp - windowing and row assembly.

#include "windows.h"

#include <algorithm>
#include <cmath>

namespace transcribe::diarize {

namespace {

double rms(const float * x, int32_t n) {
    double sum = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        sum += static_cast<double>(x[i]) * static_cast<double>(x[i]);
    }
    return std::sqrt(sum / static_cast<double>(n));
}

}  // namespace

std::vector<Window> speech_windows(const float *               pcm,
                                   int32_t                     n_samples,
                                   const WindowConfig &        config,
                                   const std::vector<Region> & regions) {
    const int32_t width = static_cast<int32_t>(config.window_seconds * static_cast<float>(config.sample_rate));
    const int32_t hop   = static_cast<int32_t>(config.hop_seconds * static_cast<float>(config.sample_rate));
    if (pcm == nullptr || n_samples <= 0 || width <= 0 || hop <= 0) {
        return {};
    }

    std::vector<Window> all;
    for (int32_t at = 0; at + width <= n_samples; at += hop) {
        all.push_back({ at, at + width });
    }
    // A recording shorter than one window is still one window: a short answer
    // is a speaker turn, and dropping it would lose the only voice there is.
    if (all.empty()) {
        all.push_back({ 0, n_samples });
    }

    if (!regions.empty()) {
        // A 10 ms mask, which is finer than any segmenter's boundaries and
        // coarse enough to test a window against by counting.
        const int32_t     step = config.sample_rate / 100;
        std::vector<bool> speech(static_cast<size_t>(n_samples / step + 1), false);
        for (const Region & r : regions) {
            const int64_t from = std::max<int64_t>(0, r.t0_ms) * config.sample_rate / 1000 / step;
            const int64_t to   = std::min<int64_t>(r.t1_ms, static_cast<int64_t>(n_samples) * 1000 / config.sample_rate)
                               * config.sample_rate / 1000 / step;
            for (int64_t i = from; i < to && i < static_cast<int64_t>(speech.size()); ++i) {
                speech[static_cast<size_t>(i)] = true;
            }
        }
        std::vector<Window> kept;
        for (const Window & w : all) {
            int32_t inside = 0, total = 0;
            for (int32_t i = w.from / step; i < w.to / step && i < static_cast<int32_t>(speech.size()); ++i) {
                ++total;
                if (speech[static_cast<size_t>(i)]) {
                    ++inside;
                }
            }
            if (total > 0 && inside * 2 >= total) {
                kept.push_back(w);
            }
        }
        return kept;
    }

    std::vector<double> level(all.size());
    for (size_t i = 0; i < all.size(); ++i) {
        level[i] = rms(pcm + all[i].from, all[i].to - all[i].from);
    }

    // Measured against a high quantile rather than the maximum, so one slammed
    // door does not silence the rest of the recording.
    std::vector<double> sorted = level;
    std::sort(sorted.begin(), sorted.end());
    const double loud  = sorted[static_cast<size_t>(0.95 * static_cast<double>(sorted.size() - 1))];
    const double floor = loud * static_cast<double>(config.silence_floor);

    std::vector<Window> kept;
    for (size_t i = 0; i < all.size(); ++i) {
        if (level[i] > floor) {
            kept.push_back(all[i]);
        }
    }
    return kept;
}

std::vector<Row> rows_from_labels(const std::vector<Window> &  windows,
                                  const std::vector<int32_t> & labels,
                                  int32_t                      sample_rate) {
    std::vector<Row> rows;
    if (windows.size() != labels.size() || windows.empty() || sample_rate <= 0) {
        return rows;
    }
    const auto ms = [sample_rate](int32_t samples) {
        return static_cast<int64_t>(std::llround(1000.0 * static_cast<double>(samples) / sample_rate));
    };

    size_t start = 0;
    for (size_t i = 1; i <= windows.size(); ++i) {
        const bool same_speaker = (i < windows.size()) && (labels[i] == labels[start]);
        // Windows overlap, so "adjacent" means the next one starts no later
        // than this one ends. A real gap is silence the windows were dropped
        // from, and the row ends there.
        const bool adjacent = (i < windows.size()) && (windows[i].from <= windows[i - 1].to);
        if (same_speaker && adjacent) {
            continue;
        }
        rows.push_back({ ms(windows[start].from), ms(windows[i - 1].to), labels[start] });
        start = i;
    }
    return rows;
}

}  // namespace transcribe::diarize
