// diarize/count.h - how many speakers are in a set of embeddings, and which
// window belongs to which.
//
// The question a distance threshold cannot answer. Agglomerative clustering
// cuts wherever it is told; asked to find the count by sweeping the cut, it
// gives a different answer at every distance, with no stable band to read the
// number off. Measured on an AMI meeting with four speakers in the reference,
// the sweep steps 6, 4, 3, 2 without ever settling.
//
// The count is in the spectrum instead. Build an affinity matrix from the
// embeddings, take the eigenvalues of its normalized Laplacian, and the
// number of speakers shows up as the largest gap between consecutive
// eigenvalues -- a graph of k well-connected groups has k eigenvalues near
// zero and a jump after them. NME-SC (Park, Han, Kumar, Narayanan 2019) adds
// the part that makes it usable without tuning: the affinity is pruned to
// each row's top p neighbours, and p is chosen by minimizing p over the
// largest eigengap, so nothing here is a constant fitted to a corpus.
//
// The same decomposition also assigns. Clustering the leading eigenvectors
// rather than the embeddings themselves is the point of spectral clustering:
// the map into eigenvector space pulls a group together whatever shape it had
// in the original space, which is what one voice recorded at two distances
// from a microphone needs.

#pragma once

#include <cstdint>
#include <vector>

namespace transcribe::diarize {

// A spectral clustering of the embeddings.
struct Spectrum {
    int32_t speakers = 1;
    // Which rows of the input were used. The eigendecomposition is cubic in
    // them, and the count is a property of the recording rather than of how
    // finely it was windowed, so a few hundred evenly spaced rows stand in
    // for all of them.
    std::vector<int32_t> sampled;
    // One 0-based label per sampled row.
    std::vector<int32_t> labels;
};

// Cluster n embeddings of `dim` floats each, row-major, into at most
// max_speakers groups. With want_speakers > 0 the count is taken as given and
// only the assignment is computed.
Spectrum spectral(const float * embeddings, int32_t n, int32_t dim, int32_t max_speakers, int32_t want_speakers);

// Just the count, for callers that assign some other way.
int32_t estimate_speakers(const float * embeddings, int32_t n, int32_t dim, int32_t max_speakers);

}  // namespace transcribe::diarize
