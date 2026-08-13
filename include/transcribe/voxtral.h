/*
 * include/transcribe/voxtral.h - Voxtral-family public extension.
 *
 * Includes transcribe.h; safe to include in C or C++ TUs. Holds the
 * free-text instruction run extension, its kind constant, and its init
 * function.
 *
 * Voxtral is an audio-LLM: a Whisper-large-v3 encoder feeding a
 * Llama/Ministral causal LM with audio-token injection. Two prompt modes
 * share the same weights:
 *
 *   transcription : [BOS][INST][BEGIN_AUDIO][AUDIO]*N[/INST](lang:<l>)?[TRANSCRIBE]
 *   instruct      : [BOS][INST][BEGIN_AUDIO][AUDIO]*N BPE(instruction)[/INST]
 *
 * Instruct mode already carries the synthesized "Translate this to X."
 * used by TRANSCRIBE_TASK_TRANSLATE. This extension is what lets a caller
 * put its own text there, which is the mechanism behind the family's
 * TRANSCRIBE_FEATURE_INITIAL_PROMPT: a free-text instruction that biases
 * the transcript toward expected vocabulary, or asks for something other
 * than a plain transcript.
 *
 * That is not Whisper's initial prompt. Whisper conditions its decoder on
 * prior-context tokens; this is an instruction to a language model, which
 * may follow it loosely or not at all, and which shares the decoder
 * context window with the audio tokens and the transcript (see
 * docs/input-limits.md - Voxtral is a hard-context-cap family). Keep it
 * short.
 *
 * Probe via transcribe_model_accepts_ext_kind(model,
 * TRANSCRIBE_EXT_SLOT_RUN, TRANSCRIBE_EXT_KIND_VOXTRAL_RUN) before
 * pointing transcribe_run_params::family at the struct.
 *
 * FourCC kinds are reserved in docs/extension-kinds.md.
 */

#ifndef TRANSCRIBE_VOXTRAL_H
#define TRANSCRIBE_VOXTRAL_H

#include "transcribe.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 'VXRN' little-endian = 0x4E525856 */
#define TRANSCRIBE_EXT_KIND_VOXTRAL_RUN 0x4E525856u

/*
 * Voxtral-family run knobs. Reached via transcribe_run_params::family.
 * NULL family selects transcription mode, the family default.
 */
struct transcribe_voxtral_run_ext {
    struct transcribe_ext ext;

    /*
     * Free-text instruction, UTF-8. NULL or "" means none, which leaves the
     * family in transcription mode exactly as passing no extension does.
     * Anything else selects instruct mode: the text is tokenized and placed
     * after the audio tokens, inside the [INST] block, exactly where the
     * synthesized translate instruction goes.
     *
     * Biasing a transcript toward known vocabulary is what this is for
     * here, e.g. "Transcribe. Expected terms: NixOS, nixpkgs, direnv."
     * The model is free to ignore it.
     *
     * Combining this with TRANSCRIBE_TASK_TRANSLATE is
     * TRANSCRIBE_ERR_INVALID_ARG: both want the one instruction slot, and
     * silently dropping either would be worse than refusing.
     */
    const char * instruction;
};

/* Fills ext.size/kind and instruction = NULL (transcription mode). */
TRANSCRIBE_API void transcribe_voxtral_run_ext_init(struct transcribe_voxtral_run_ext * ext);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TRANSCRIBE_VOXTRAL_H */
