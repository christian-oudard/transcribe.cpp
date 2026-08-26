// diarize/count.cpp - speaker counting and assignment by the normalized
// maximum eigengap.

#include "count.h"

#include "transcribe-log.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

namespace transcribe::diarize {

namespace {

// Rows kept for the spectrum. The count does not get better with more
// windows, and the eigendecomposition is cubic in them.
constexpr int32_t kMaxRows = 400;

// Pruning widths tried: NME sweeps p and keeps the one that minimizes p over
// the eigengap it produces. A coarse sweep is enough -- the criterion is flat
// between neighbouring widths and every candidate costs a decomposition.
constexpr int kSweep = 12;

// Unit-length copy of an evenly spaced sample of the embeddings.
std::vector<float> sample(const float *          embeddings,
                          int32_t                n,
                          int32_t                dim,
                          std::vector<int32_t> & taken) {
    const int32_t rows = std::min(n, kMaxRows);
    taken.resize(static_cast<size_t>(rows));
    std::vector<float> out(static_cast<size_t>(rows) * dim);
    for (int32_t i = 0; i < rows; ++i) {
        const int32_t src              = static_cast<int32_t>(static_cast<int64_t>(i) * n / rows);
        taken[static_cast<size_t>(i)]  = src;
        const float * v                = embeddings + static_cast<size_t>(src) * dim;
        double        sum              = 0.0;
        for (int32_t k = 0; k < dim; ++k) {
            sum += static_cast<double>(v[k]) * v[k];
        }
        const double inv = (sum > 0.0) ? 1.0 / std::sqrt(sum) : 0.0;
        for (int32_t k = 0; k < dim; ++k) {
            out[static_cast<size_t>(i) * dim + k] = static_cast<float>(v[k] * inv);
        }
    }
    return out;
}

// Symmetric eigendecomposition by the cyclic Jacobi method: eigenvalues
// ascending in `ev`, the matching eigenvectors as the columns of `vec`.
//
// Small and exact enough. At a few hundred rows it costs milliseconds, and
// unlike an iterative solver it needs no starting guess and cannot miss a
// cluster of close eigenvalues -- which is the very thing being measured.
void jacobi(std::vector<double> & a, int n, std::vector<double> & ev, std::vector<double> & vec) {
    vec.assign(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) {
        vec[static_cast<size_t>(i) * n + i] = 1.0;
    }
    for (int sweep = 0; sweep < 30; ++sweep) {
        double off = 0.0;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                off += a[static_cast<size_t>(i) * n + j] * a[static_cast<size_t>(i) * n + j];
            }
        }
        if (off < 1e-18) {
            break;
        }
        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const double apq = a[static_cast<size_t>(p) * n + q];
                if (std::fabs(apq) < 1e-15) {
                    continue;
                }
                const double app   = a[static_cast<size_t>(p) * n + p];
                const double aqq   = a[static_cast<size_t>(q) * n + q];
                const double theta = 0.5 * (aqq - app) / apq;
                const double t     = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c     = 1.0 / std::sqrt(t * t + 1.0);
                const double s     = t * c;
                for (int k = 0; k < n; ++k) {
                    const double akp                  = a[static_cast<size_t>(k) * n + p];
                    const double akq                  = a[static_cast<size_t>(k) * n + q];
                    a[static_cast<size_t>(k) * n + p] = c * akp - s * akq;
                    a[static_cast<size_t>(k) * n + q] = s * akp + c * akq;
                }
                for (int k = 0; k < n; ++k) {
                    const double apk                  = a[static_cast<size_t>(p) * n + k];
                    const double aqk                  = a[static_cast<size_t>(q) * n + k];
                    a[static_cast<size_t>(p) * n + k] = c * apk - s * aqk;
                    a[static_cast<size_t>(q) * n + k] = s * apk + c * aqk;
                    const double vkp                  = vec[static_cast<size_t>(k) * n + p];
                    const double vkq                  = vec[static_cast<size_t>(k) * n + q];
                    vec[static_cast<size_t>(k) * n + p] = c * vkp - s * vkq;
                    vec[static_cast<size_t>(k) * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    ev.assign(static_cast<size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
        ev[static_cast<size_t>(i)] = a[static_cast<size_t>(i) * n + i];
    }
}

// Cosine affinity, negatives clipped away: two voices pointing apart are
// unrelated rather than related in reverse.
std::vector<double> affinity_of(const std::vector<float> & e, int32_t rows, int32_t dim) {
    std::vector<double> a(static_cast<size_t>(rows) * rows);
    for (int32_t i = 0; i < rows; ++i) {
        for (int32_t j = 0; j < rows; ++j) {
            double dot = 0.0;
            for (int32_t k = 0; k < dim; ++k) {
                dot += static_cast<double>(e[static_cast<size_t>(i) * dim + k]) * e[static_cast<size_t>(j) * dim + k];
            }
            a[static_cast<size_t>(i) * rows + j] = (dot > 0.0) ? dot : 0.0;
        }
    }
    return a;
}

// Normalized Laplacian of the affinity pruned to each row's top p neighbours.
// An edge survives if either end considers the other a neighbour.
std::vector<double> laplacian(const std::vector<double> & affinity, int32_t rows, int32_t p) {
    std::vector<double> w(static_cast<size_t>(rows) * rows, 0.0);
    std::vector<double> row;
    for (int32_t i = 0; i < rows; ++i) {
        row.assign(affinity.begin() + static_cast<size_t>(i) * rows,
                   affinity.begin() + static_cast<size_t>(i + 1) * rows);
        std::nth_element(row.begin(), row.begin() + p, row.end(), std::greater<double>());
        const double keep = row[static_cast<size_t>(p)];
        for (int32_t j = 0; j < rows; ++j) {
            if (i != j && affinity[static_cast<size_t>(i) * rows + j] >= keep) {
                w[static_cast<size_t>(i) * rows + j] = affinity[static_cast<size_t>(i) * rows + j];
            }
        }
    }
    for (int32_t i = 0; i < rows; ++i) {
        for (int32_t j = i + 1; j < rows; ++j) {
            const double v                       = std::max(w[static_cast<size_t>(i) * rows + j],
                                                            w[static_cast<size_t>(j) * rows + i]);
            w[static_cast<size_t>(i) * rows + j] = v;
            w[static_cast<size_t>(j) * rows + i] = v;
        }
    }
    std::vector<double> deg(static_cast<size_t>(rows), 0.0);
    for (int32_t i = 0; i < rows; ++i) {
        for (int32_t j = 0; j < rows; ++j) {
            deg[static_cast<size_t>(i)] += w[static_cast<size_t>(i) * rows + j];
        }
    }
    std::vector<double> lap(static_cast<size_t>(rows) * rows, 0.0);
    for (int32_t i = 0; i < rows; ++i) {
        const double di = (deg[static_cast<size_t>(i)] > 0.0) ? 1.0 / std::sqrt(deg[static_cast<size_t>(i)]) : 0.0;
        for (int32_t j = 0; j < rows; ++j) {
            const double dj = (deg[static_cast<size_t>(j)] > 0.0) ? 1.0 / std::sqrt(deg[static_cast<size_t>(j)]) : 0.0;
            lap[static_cast<size_t>(i) * rows + j] =
                (i == j ? 1.0 : 0.0) - w[static_cast<size_t>(i) * rows + j] * di * dj;
        }
    }
    return lap;
}

// k-means over the spectral embedding, seeded farthest-point-first. No
// randomness: the same recording clusters the same way twice, which matters
// more here than escaping a bad local minimum, and the spectral map leaves
// the groups round enough that the seeding is not delicate.
std::vector<int32_t> kmeans(const std::vector<double> & x, int32_t rows, int32_t k) {
    std::vector<int32_t> label(static_cast<size_t>(rows), 0);
    if (k <= 1) {
        return label;
    }
    std::vector<double> centre(static_cast<size_t>(k) * k, 0.0);
    const auto          dist2 = [&](int32_t i, const double * c) {
        double d = 0.0;
        for (int32_t f = 0; f < k; ++f) {
            const double v = x[static_cast<size_t>(i) * k + f] - c[f];
            d += v * v;
        }
        return d;
    };

    // Seed 0 is the row furthest from the mean; each later seed is the row
    // furthest from every seed so far.
    std::vector<double> mean(static_cast<size_t>(k), 0.0);
    for (int32_t i = 0; i < rows; ++i) {
        for (int32_t f = 0; f < k; ++f) {
            mean[static_cast<size_t>(f)] += x[static_cast<size_t>(i) * k + f] / rows;
        }
    }
    std::vector<int32_t> seed;
    {
        int32_t at   = 0;
        double  best = -1.0;
        for (int32_t i = 0; i < rows; ++i) {
            const double d = dist2(i, mean.data());
            if (d > best) {
                best = d;
                at   = i;
            }
        }
        seed.push_back(at);
    }
    while (static_cast<int32_t>(seed.size()) < k) {
        int32_t at   = 0;
        double  best = -1.0;
        for (int32_t i = 0; i < rows; ++i) {
            double near = 1e300;
            for (int32_t s : seed) {
                near = std::min(near, dist2(i, &x[static_cast<size_t>(s) * k]));
            }
            if (near > best) {
                best = near;
                at   = i;
            }
        }
        seed.push_back(at);
    }
    for (int32_t c = 0; c < k; ++c) {
        for (int32_t f = 0; f < k; ++f) {
            centre[static_cast<size_t>(c) * k + f] = x[static_cast<size_t>(seed[static_cast<size_t>(c)]) * k + f];
        }
    }

    for (int iter = 0; iter < 50; ++iter) {
        bool moved = false;
        for (int32_t i = 0; i < rows; ++i) {
            int32_t at   = 0;
            double  best = 1e300;
            for (int32_t c = 0; c < k; ++c) {
                const double d = dist2(i, &centre[static_cast<size_t>(c) * k]);
                if (d < best) {
                    best = d;
                    at   = c;
                }
            }
            if (label[static_cast<size_t>(i)] != at) {
                label[static_cast<size_t>(i)] = at;
                moved                         = true;
            }
        }
        if (!moved && iter > 0) {
            break;
        }
        std::vector<double>  sum(static_cast<size_t>(k) * k, 0.0);
        std::vector<int32_t> count(static_cast<size_t>(k), 0);
        for (int32_t i = 0; i < rows; ++i) {
            const int32_t c = label[static_cast<size_t>(i)];
            ++count[static_cast<size_t>(c)];
            for (int32_t f = 0; f < k; ++f) {
                sum[static_cast<size_t>(c) * k + f] += x[static_cast<size_t>(i) * k + f];
            }
        }
        for (int32_t c = 0; c < k; ++c) {
            if (count[static_cast<size_t>(c)] == 0) {
                continue;  // An empty group keeps its seed rather than moving to the origin.
            }
            for (int32_t f = 0; f < k; ++f) {
                centre[static_cast<size_t>(c) * k + f] =
                    sum[static_cast<size_t>(c) * k + f] / count[static_cast<size_t>(c)];
            }
        }
    }
    return label;
}


}  // namespace

// Below this an eigenvalue of the normalized Laplacian is zero rather than
// small, which is what says the graph is in more than one piece.
constexpr double kZeroEigen = 1e-6;

Spectrum spectral(const float * embeddings, int32_t n, int32_t dim, int32_t max_speakers, int32_t want_speakers) {
    Spectrum out;
    if (embeddings == nullptr || n <= 1 || dim <= 0 || max_speakers < 1) {
        out.sampled = { 0 };
        out.labels  = { 0 };
        return out;
    }
    const std::vector<float> e    = sample(embeddings, n, dim, out.sampled);
    const int32_t            rows = static_cast<int32_t>(out.sampled.size());
    if (rows < 4) {
        out.labels.assign(static_cast<size_t>(rows), 0);
        return out;
    }

    const std::vector<double> affinity = affinity_of(e, rows, dim);
    const int32_t             cap      = std::min(max_speakers, rows - 2);

    int32_t best_p          = 2;
    int32_t best_k          = 1;
    double  best_ratio      = 0.0;
    int32_t best_components = rows + 1;
    for (int step = 1; step <= kSweep; ++step) {
        // Widths spread across the plausible range: too few neighbours and the
        // graph falls apart into isolated windows, too many and everyone is
        // connected to everyone.
        const int32_t p = std::max(2, static_cast<int32_t>(static_cast<double>(rows) * step / (kSweep * 4)));
        if (p >= rows) {
            break;
        }
        std::vector<double> lap = laplacian(affinity, rows, p);
        std::vector<double> ev, vec;
        jacobi(lap, rows, ev, vec);
        std::sort(ev.begin(), ev.end());

        int32_t k   = 1;
        double  gap = 0.0;
        for (int32_t i = 0; i < cap; ++i) {
            const double g = ev[static_cast<size_t>(i + 1)] - ev[static_cast<size_t>(i)];
            if (g > gap) {
                gap = g;
                k   = i + 1;
            }
        }
        // How many pieces the pruned graph fell into. For a normalized
        // Laplacian the multiplicity of eigenvalue zero is exactly the number
        // of connected components, and a component is not a speaker: keeping
        // only two neighbours per window on a short recording disconnects it,
        // and the widest eigengap then measures how badly it broke.
        //
        // The boundary is not delicate. Jacobi returns a true zero at around
        // 1e-12, and the smallest non-zero eigenvalue seen on a connected
        // graph here is 0.019.
        int32_t components = 0;
        for (const double v : ev) {
            if (v < kZeroEigen) {
                ++components;
            }
        }

        // The pruning width to believe is the one whose eigengap is widest for
        // its size, which is what makes this need no tuned constant -- but
        // only among widths that left the graph in one piece. Spectral
        // clustering counts clusters in a connected graph; on a disconnected
        // one the count it reads is the number of pieces, which is a fact
        // about the pruning and not about the room.
        const double ratio = gap / static_cast<double>(p);
        log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG,
                "diarize: p=%d k=%d gap=%.4f ratio=%.5f parts=%d ev=%.3f %.3f %.3f %.3f %.3f %.3f", p, k, gap, ratio,
                components, ev[0], ev[1], ev[2], ev[3], ev[4], ev[5]);
        // Fewest pieces first, so a connected graph beats a broken one however
        // wide the broken one's gap. Every width disconnecting is possible --
        // very short audio, or a room where nobody sounds like anybody -- and
        // then the least broken one is the best there is.
        if (components < best_components || (components == best_components && ratio > best_ratio)) {
            best_components = components;
            best_ratio      = ratio;
            best_k          = k;
            best_p          = p;
        }
    }

    out.speakers = (want_speakers > 0) ? std::min(want_speakers, rows) : std::max(1, std::min(best_k, max_speakers));

    // Assign in the space the count was read from. The eigenvectors of the
    // smallest eigenvalues are where a speaker's windows sit together whatever
    // shape they had as embeddings, which is the whole reason to decompose.
    std::vector<double> lap = laplacian(affinity, rows, best_p);
    std::vector<double> ev, vec;
    jacobi(lap, rows, ev, vec);

    std::vector<int32_t> order(static_cast<size_t>(rows));
    for (int32_t i = 0; i < rows; ++i) {
        order[static_cast<size_t>(i)] = i;
    }
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
        return ev[static_cast<size_t>(a)] < ev[static_cast<size_t>(b)];
    });

    const int32_t       k = out.speakers;
    std::vector<double> x(static_cast<size_t>(rows) * k, 0.0);
    for (int32_t i = 0; i < rows; ++i) {
        double norm = 0.0;
        for (int32_t f = 0; f < k; ++f) {
            const double v                     = vec[static_cast<size_t>(i) * rows + order[static_cast<size_t>(f)]];
            x[static_cast<size_t>(i) * k + f]  = v;
            norm += v * v;
        }
        // Row-normalized spectral embedding (Ng, Jordan, Weiss): what matters
        // is the direction in eigenvector space, not how strongly this window
        // was connected to anything.
        norm = std::sqrt(norm);
        if (norm > 0.0) {
            for (int32_t f = 0; f < k; ++f) {
                x[static_cast<size_t>(i) * k + f] /= norm;
            }
        }
    }
    out.labels = kmeans(x, rows, k);
    return out;
}

int32_t estimate_speakers(const float * embeddings, int32_t n, int32_t dim, int32_t max_speakers) {
    return spectral(embeddings, n, dim, max_speakers, 0).speakers;
}

}  // namespace transcribe::diarize
