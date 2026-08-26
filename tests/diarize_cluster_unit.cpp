// Unit test: cosine agglomerative clustering of speaker embeddings.
//
// Synthetic vectors rather than a model's: what is being tested is the metric,
// the two cut policies and the numbering, none of which know what a voice is.

#include "diarize/cluster.h"

#include "diarize/count.h"

#include <cmath>
#include <set>
#include <cstdio>
#include <vector>

using transcribe::diarize::ClusterConfig;
using transcribe::diarize::cluster;
using transcribe::diarize::estimate_speakers;

// The cut-at-a-distance path. Most of what follows tests the metric and the
// cut policy, which is a different question from how many speakers there are,
// so those cases ask for the distance explicitly.
ClusterConfig by_distance() {
    ClusterConfig cfg;
    cfg.estimate_count = false;
    return cfg;
}

namespace {

int failures = 0;

void check(bool ok, const char * what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

// A cluster of `n` vectors around direction `dir` in a `dim`-dimensional
// space, each nudged along a different axis so they are close but not equal.
void add_group(std::vector<float> & out, int dim, int dir, int n, float spread) {
    for (int i = 0; i < n; ++i) {
        std::vector<float> v(static_cast<size_t>(dim), 0.0f);
        v[static_cast<size_t>(dir)] = 1.0f;
        v[static_cast<size_t>((dir + 1 + i) % dim)] += spread;
        out.insert(out.end(), v.begin(), v.end());
    }
}

}  // namespace

int main() {
    const int dim = 8;

    // Three tight groups, far apart. Anything that works at all finds three.
    {
        std::vector<float> emb;
        add_group(emb, dim, 0, 4, 0.05f);
        add_group(emb, dim, 3, 3, 0.05f);
        add_group(emb, dim, 6, 3, 0.05f);
        const int n = 10;

        std::vector<int32_t> labels = cluster(emb.data(), n, dim, by_distance());
        check(labels.size() == 10, "one label per row");
        check(labels[0] == labels[1] && labels[1] == labels[2] && labels[2] == labels[3], "first group holds together");
        check(labels[4] == labels[5] && labels[5] == labels[6], "second group holds together");
        check(labels[0] != labels[4] && labels[4] != labels[7] && labels[0] != labels[7], "the groups stay apart");
    }

    // A speaker count overrides the distance: the same vectors cut into two
    // groups when two is what the caller says there are.
    {
        std::vector<float> emb;
        add_group(emb, dim, 0, 4, 0.05f);
        add_group(emb, dim, 3, 3, 0.05f);
        add_group(emb, dim, 6, 3, 0.05f);

        ClusterConfig cfg = by_distance();
        cfg.num_speakers  = 2;
        std::vector<int32_t> labels = cluster(emb.data(), 10, dim, cfg);
        int                  seen[3] = { 0, 0, 0 };
        for (int32_t l : labels) {
            check(l >= 0 && l < 2, "labels stay inside the requested count");
            if (l >= 0 && l < 2) {
                ++seen[l];
            }
        }
        check(seen[0] > 0 && seen[1] > 0, "both requested speakers are used");
    }

    // Numbering follows first appearance, so a transcript opens on S00.
    {
        std::vector<float> emb;
        add_group(emb, dim, 5, 2, 0.05f);
        add_group(emb, dim, 1, 2, 0.05f);
        std::vector<int32_t> labels = cluster(emb.data(), 4, dim, by_distance());
        check(labels[0] == 0, "the first window is speaker 0");
        check(labels[2] == 1, "the next voice is speaker 1");
    }

    // One window is one speaker, and no windows is no speakers. Both happen on
    // real audio: a recording with a single short utterance in it, and one the
    // segmenter found no speech in at all.
    {
        std::vector<float> one(static_cast<size_t>(dim), 0.0f);
        one[0]                          = 1.0f;
        std::vector<int32_t> single     = cluster(one.data(), 1, dim, by_distance());
        check(single.size() == 1 && single[0] == 0, "a single window is speaker 0");
        check(cluster(one.data(), 0, dim, by_distance()).empty(), "no windows, no labels");
    }

    // A silent window embeds to the zero vector, which must not become a NaN
    // distance and take the whole clustering with it.
    {
        std::vector<float> emb;
        add_group(emb, dim, 0, 3, 0.05f);
        emb.insert(emb.end(), static_cast<size_t>(dim), 0.0f);
        std::vector<int32_t> labels = cluster(emb.data(), 4, dim, by_distance());
        check(labels.size() == 4, "the silent window still gets a label");
        check(labels[3] != labels[0], "silence is not one of the speakers");
    }

    // Counting, which is the question a distance cannot answer. Three groups
    // of twenty, each around its own axis.
    {
        std::vector<float> emb;
        for (int g = 0; g < 3; ++g) {
            add_group(emb, dim, g * 2, 20, 0.04f);
        }
        const int32_t k = estimate_speakers(emb.data(), 60, dim, 10);
        check(k == 3, "three separated groups are counted as three speakers");

        std::vector<int32_t> labels = cluster(emb.data(), 60, dim, ClusterConfig{});
        check(static_cast<int32_t>(std::set<int32_t>(labels.begin(), labels.end()).size()) == 3,
              "and clustering with no count given finds three");
    }

    if (failures == 0) {
        std::printf("diarize_cluster_unit: OK\n");
    }
    return failures == 0 ? 0 : 1;
}
