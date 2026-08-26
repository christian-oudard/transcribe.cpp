/*
 * include/transcribe/titanet.h - TitaNet-family public extension.
 *
 * Includes transcribe.h; safe to include in C or C++ TUs. Holds the
 * diarization run extension, its kind constant, and its init function.
 *
 * TitaNet is a speaker embedding model: a run produces no text. With
 * transcribe_run_params::diarize on, the recording is cut into windows, each
 * window is embedded, and windows whose embeddings point the same way are
 * clustered into one speaker; the product is the who-spoke-when rows read
 * back via transcribe_n_speaker_segments / transcribe_get_speaker_segment
 * (TRANSCRIBE_FEATURE_DIARIZATION).
 *
 * This extension answers the one question the audio cannot: how many people
 * are in it. Give a count and the clustering cuts at exactly that many
 * speakers. Give none and it cuts at a distance instead, which is a property
 * of the embedding model rather than of the recording -- so a caller who
 * knows the count should say so, and one who does not should expect the
 * number to be approximate.
 *
 * Probe via transcribe_model_accepts_ext_kind(model,
 * TRANSCRIBE_EXT_SLOT_RUN, TRANSCRIBE_EXT_KIND_TITANET_DIARIZE) before
 * pointing transcribe_run_params::family at the struct.
 *
 * FourCC kinds are reserved in docs/extension-kinds.md.
 */

#ifndef TRANSCRIBE_TITANET_H
#define TRANSCRIBE_TITANET_H

#include "transcribe.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 'TNDZ' little-endian = 0x5A444E54 */
#define TRANSCRIBE_EXT_KIND_TITANET_DIARIZE 0x5A444E54u

struct transcribe_titanet_diarize_ext {
    struct transcribe_ext ext;

    /*
     * Exact number of speakers, when it is known. 0 leaves the count to the
     * threshold below. Negative is rejected.
     *
     * Worth supplying wherever possible: a count is a fact about the room,
     * and every alternative is an inference from distances between voices.
     */
    int32_t num_speakers;

    /*
     * Cosine distance at which two windows stop being the same person, used
     * only when num_speakers is 0. 0 takes the family default; the useful
     * range is roughly 0.5 (splits one speaker into several) to 0.9 (merges
     * several into one). Outside [0, 2] is rejected, since cosine distance
     * cannot leave that interval.
     */
    float threshold;
};

/* Stamps the header (size + kind) and clears both fields to their
 * defaults. Call before setting any field. */
void transcribe_titanet_diarize_ext_init(struct transcribe_titanet_diarize_ext * p);

#ifdef __cplusplus
}
#endif

#endif /* TRANSCRIBE_TITANET_H */
