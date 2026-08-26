// diarize/cluster.cpp - cosine agglomerative clustering over speaker
// embeddings, on top of the vendored fastcluster.

#include "cluster.h"

#include "count.h"

#include "third_party/fastcluster/fastcluster.h"

#include <cmath>
#include <vector>

namespace transcribe::diarize {

namespace {

// Average linkage rather than complete. Complete linkage asks that every pair
// in a group be close, which splits one person who was recorded at two
// distances from the microphone; average asks that the group be close on the
// whole, which is the shape a voice actually has. It is also what pyannote
// and NeMo cut with.
constexpr int kLinkage = HCLUST_METHOD_AVERAGE;

// Unit-length copy, so the distance below is a pure angle.
std::vector<float> normalized(const float * embeddings, int32_t n, int32_t dim) {
    std::vector<float> out(static_cast<size_t>(n) * static_cast<size_t>(dim));
    for (int32_t i = 0; i < n; ++i) {
        const float * src = embeddings + static_cast<size_t>(i) * dim;
        float *       dst = out.data() + static_cast<size_t>(i) * dim;
        double        sum = 0.0;
        for (int32_t k = 0; k < dim; ++k) {
            sum += static_cast<double>(src[k]) * static_cast<double>(src[k]);
        }
        // A window with no energy in it embeds to nothing. Leaving it at zero
        // keeps it at distance 1 from everything, which is where a vector
        // carrying no voice belongs.
        const double norm = std::sqrt(sum);
        const double inv  = (norm > 0.0) ? 1.0 / norm : 0.0;
        for (int32_t k = 0; k < dim; ++k) {
            dst[k] = static_cast<float>(static_cast<double>(src[k]) * inv);
        }
    }
    return out;
}

// Renumber so speakers appear in the order they first speak. Nothing
// downstream depends on it, and a transcript whose first voice is S03 reads
// like a mistake.
void renumber_by_appearance(std::vector<int32_t> & labels) {
    std::vector<int32_t> seen(labels.size(), -1);
    int32_t              next = 0;
    for (int32_t & label : labels) {
        if (seen[static_cast<size_t>(label)] < 0) {
            seen[static_cast<size_t>(label)] = next++;
        }
        label = seen[static_cast<size_t>(label)];
    }
}

// Assign every row to the nearest of the spectral clusters, by cosine to the
// cluster's mean embedding.
//
// The decomposition runs on a few hundred sampled rows because it is cubic in
// them; the rest of the recording is attributed by which of the groups it
// sounds most like, which is cheap and does not need the sample to have seen
// that particular window.
std::vector<int32_t> by_nearest_centre(const std::vector<float> & unit,
                                       int32_t                    n,
                                       int32_t                    dim,
                                       const Spectrum &           s) {
    const int32_t      k = std::max(1, s.speakers);
    std::vector<double> centre(static_cast<size_t>(k) * dim, 0.0);
    std::vector<int32_t> held(static_cast<size_t>(k), 0);
    for (size_t i = 0; i < s.sampled.size(); ++i) {
        const int32_t c = s.labels[i];
        if (c < 0 || c >= k) {
            continue;
        }
        ++held[static_cast<size_t>(c)];
        const float * v = unit.data() + static_cast<size_t>(s.sampled[i]) * dim;
        for (int32_t f = 0; f < dim; ++f) {
            centre[static_cast<size_t>(c) * dim + f] += v[f];
        }
    }
    for (int32_t c = 0; c < k; ++c) {
        double norm = 0.0;
        for (int32_t f = 0; f < dim; ++f) {
            const double v = centre[static_cast<size_t>(c) * dim + f];
            norm += v * v;
        }
        norm = std::sqrt(norm);
        if (norm > 0.0) {
            for (int32_t f = 0; f < dim; ++f) {
                centre[static_cast<size_t>(c) * dim + f] /= norm;
            }
        }
    }

    std::vector<int32_t> labels(static_cast<size_t>(n), 0);
    for (int32_t i = 0; i < n; ++i) {
        const float * v    = unit.data() + static_cast<size_t>(i) * dim;
        int32_t       at   = 0;
        double        best = -2.0;
        for (int32_t c = 0; c < k; ++c) {
            if (held[static_cast<size_t>(c)] == 0) {
                continue;
            }
            double dot = 0.0;
            for (int32_t f = 0; f < dim; ++f) {
                dot += static_cast<double>(v[f]) * centre[static_cast<size_t>(c) * dim + f];
            }
            if (dot > best) {
                best = dot;
                at   = c;
            }
        }
        labels[static_cast<size_t>(i)] = at;
    }
    return labels;
}

}  // namespace

std::vector<int32_t> cluster(const float * embeddings, int32_t n, int32_t dim, const ClusterConfig & config) {
    if (embeddings == nullptr || n <= 0 || dim <= 0) {
        return {};
    }
    if (n == 1) {
        return { 0 };
    }

    const std::vector<float> unit = normalized(embeddings, n, dim);

    // The spectral path, which is the default: it decides the count and does
    // the assignment in the same space. The agglomerative path below is what
    // is left for a caller who supplies a distance instead.
    if (config.num_speakers > 0 || config.estimate_count) {
        const Spectrum s      = spectral(embeddings, n, dim, config.max_speakers, config.num_speakers);
        std::vector<int32_t> labels = by_nearest_centre(unit, n, dim, s);
        renumber_by_appearance(labels);
        return labels;
    }


    // Condensed upper triangle, the layout fastcluster takes: pair (i, j) for
    // i < j, in row order.
    std::vector<double> distance(static_cast<size_t>(n) * static_cast<size_t>(n - 1) / 2);
    size_t              at = 0;
    for (int32_t i = 0; i < n; ++i) {
        const float * a = unit.data() + static_cast<size_t>(i) * dim;
        for (int32_t j = i + 1; j < n; ++j) {
            const float * b   = unit.data() + static_cast<size_t>(j) * dim;
            double        dot = 0.0;
            for (int32_t k = 0; k < dim; ++k) {
                dot += static_cast<double>(a[k]) * static_cast<double>(b[k]);
            }
            // Both are unit length, so this is 1 - cos(angle), in [0, 2].
            // Clamped because rounding can put an identical pair a hair below
            // zero, and a negative distance is not one.
            const double d = 1.0 - dot;
            distance[at++] = (d > 0.0) ? d : 0.0;
        }
    }

    std::vector<int32_t> merge(static_cast<size_t>(2 * (n - 1)));
    std::vector<double>  height(static_cast<size_t>(n - 1));
    hclust_fast(n, distance.data(), kLinkage, merge.data(), height.data());

    std::vector<int32_t> labels(static_cast<size_t>(n));
    cutree_cdist(n, merge.data(), height.data(), config.threshold, labels.data());
    renumber_by_appearance(labels);
    return labels;
}

}  // namespace transcribe::diarize
