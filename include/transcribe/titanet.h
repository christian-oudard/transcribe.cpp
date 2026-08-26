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

    /*
     * Where the speech is, as pairs of milliseconds (start, end), and how many
     * pairs. NULL leaves the model to find speech by loudness alone.
     *
     * Worth supplying, and the reason is measured. Loudness cannot tell a
     * voice from a chair or a keyboard: on two AMI meetings it passed 97 and
     * 172 seconds of audible non-speech, which then clustered coherently and
     * arrived as an extra speaker in a room of four. Regions from anything
     * that decides speech on more than energy -- a voice activity detector, a
     * segmenter, or the word timings of a transcriber that has already run
     * over the same audio -- remove that cluster.
     *
     * A window is embedded when at least half of it falls inside a region.
     * Regions need not be sorted and may overlap.
     */
    const int64_t * speech_ms;
    int32_t         n_speech;
};

/* Stamps the header (size + kind) and clears both fields to their
 * defaults. Call before setting any field. */
TRANSCRIBE_API void transcribe_titanet_diarize_ext_init(struct transcribe_titanet_diarize_ext * p);

#ifdef __cplusplus
}
#endif

#endif /* TRANSCRIBE_TITANET_H */
