// diarize/cluster.h - group speaker embeddings by whose voice they are.
//
// The last stage of diarization and the only one that is not a neural
// network. Everything before it produces one vector per window of speech;
// this decides which windows belong to the same person, which is what turns
// vectors into speaker labels.
//
// Two ways to stop, and the difference matters more than the algorithm. Given
// a speaker count, the clustering cuts its tree at exactly that many groups
// and cannot be wrong about the number. Given no count, it cuts at a distance
// instead, and that distance is a property of the embedding model rather than
// of the audio -- so a caller who knows how many people are in the room
// should say so.
//
// The clustering itself is vendored (src/third_party/fastcluster); this is
// the cosine metric, the condensed distance matrix and the two cut policies.

#pragma once

#include <cstdint>
#include <vector>

namespace transcribe::diarize {

// How to decide the number of speakers.
struct ClusterConfig {
    // Exact number of speakers, when the caller knows it. Takes precedence
    // over everything below.
    int32_t num_speakers = 0;

    // With no count given, estimate one from the spectrum (see count.h)
    // rather than cutting at a distance. On by default because a threshold
    // cannot count: on an AMI meeting with four speakers the distance sweep
    // steps 6, 4, 3, 2 with no stable band, where the eigengap says four.
    bool estimate_count = true;

    // The most speakers an estimate may return. Not a property of the audio,
    // a bound on the search.
    int32_t max_speakers = 20;

    // The fallback when estimate_count is off: the cosine distance at which two windows stop being the same
    // person. 1 - cosine similarity, so 0 is identical and 1 is orthogonal.
    //
    // The default is the middle of the plateau measured over two known
    // speakers at the window length in diarize/windows.h: every value from
    // 0.55 to 0.80 recovered exactly the right speakers there, so 0.7 is the
    // furthest from being wrong in either direction.
    //
    // It is still a constant about voices in general rather than about these
    // voices. A room of similar voices wants more and a recording of one
    // speaker among many wants less, and neither is knowable from here --
    // which is why num_speakers is worth asking the caller for.
    float threshold = 0.70f;
};

// Assign each row of `embeddings` (n rows of `dim` floats, row-major) a
// speaker index, numbered from 0 in order of first appearance.
//
// Embeddings do not have to be normalized; this normalizes its own copy, so
// only direction counts, which is what a speaker embedding means.
//
// Returns one label per row. An empty input returns empty; a single row is
// one speaker, since there is nothing for it to be distant from.
std::vector<int32_t> cluster(const float * embeddings, int32_t n, int32_t dim, const ClusterConfig & config);

}  // namespace transcribe::diarize
