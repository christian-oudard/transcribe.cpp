// voxtral_run_ext_unit.cpp - Voxtral free-text instruction run extension
// (TRANSCRIBE_EXT_KIND_VOXTRAL_RUN, RUN slot).
//
// Covers, against a real GGUF (env-gated, RC 77 skip):
//
//   1. transcribe_model_accepts_ext_kind: VXRN accepted on _RUN only;
//      foreign kinds and the _STREAM slot rejected.
//   2. transcribe_voxtral_run_ext_init stamps size/kind and leaves the
//      instruction NULL, which is transcription mode.
//   3. Pre-clear rejection: a wrong-kind ext, and an instruction combined
//      with TASK_TRANSLATE, both fail with INVALID_ARG and PRESERVE the
//      previous result (run_validate fires before clear_result).
//   4. A NULL or empty instruction transcribes exactly as no ext at all,
//      so the extension costs nothing when unused.
//   5. An instruction naming expected vocabulary is accepted and still
//      produces a transcript. The model is a language model and may follow
//      the instruction loosely, so this asserts that the path runs rather
//      than asserting on wording.
//
// Gated by TRANSCRIBE_VOXTRAL_GGUF, as the other real-model tests are:
// the instruction is tokenized into the prompt inside run(), so it needs
// real weights to exercise.

#include "transcribe.h"
#include "transcribe/voxtral.h"
#include "transcribe/whisper.h"
#include "wav.h"

#include <sys/stat.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

bool file_exists(const std::string & path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0;
}

// The transcript of the most recent run, for comparing paths.
std::string full_text(transcribe_session * s) {
    const char * t = transcribe_full_text(s);
    return t != nullptr ? std::string(t) : std::string();
}

}  // namespace

int main() {
    const char * gguf = std::getenv("TRANSCRIBE_VOXTRAL_GGUF");
    if (gguf == nullptr || !file_exists(gguf)) {
        std::fprintf(stderr, "skip: set TRANSCRIBE_VOXTRAL_GGUF to a Voxtral .gguf\n");
        return 77;
    }

    const std::string  wav_path = std::string(TRANSCRIBE_TEST_SAMPLES_DIR) + "/jfk.wav";
    std::vector<float> pcm;
    std::string        wav_err;
    if (!transcribe_cli::load_wav_mono_16k(wav_path, pcm, wav_err)) {
        std::fprintf(stderr, "skip: wav load: %s\n", wav_err.c_str());
        return 77;
    }

    struct transcribe_model_load_params lp{};
    transcribe_model_load_params_init(&lp);
    struct transcribe_model * model = nullptr;
    if (transcribe_model_load_file(gguf, &lp, &model) != TRANSCRIBE_OK || model == nullptr) {
        std::fprintf(stderr, "skip: cannot load %s\n", gguf);
        return 77;
    }

    // ---- 1. Kind and slot probing --------------------------------------
    CHECK(transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_RUN, TRANSCRIBE_EXT_KIND_VOXTRAL_RUN));
    // The same kind on the wrong slot, and a foreign kind on the right one.
    CHECK(!transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_STREAM, TRANSCRIBE_EXT_KIND_VOXTRAL_RUN));
    CHECK(!transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_RUN, TRANSCRIBE_EXT_KIND_WHISPER_RUN));

    // ---- 2. init stamps the header and defaults to transcription -------
    struct transcribe_voxtral_run_ext ext{};
    transcribe_voxtral_run_ext_init(&ext);
    CHECK(ext.ext.size == sizeof(ext));
    CHECK(ext.ext.kind == TRANSCRIBE_EXT_KIND_VOXTRAL_RUN);
    CHECK(ext.instruction == nullptr);

    struct transcribe_session_params sp{};
    transcribe_session_params_init(&sp);
    struct transcribe_session * sess = nullptr;
    if (transcribe_session_init(model, &sp, &sess) != TRANSCRIBE_OK || sess == nullptr) {
        std::fprintf(stderr, "skip: cannot open a session\n");
        transcribe_model_free(model);
        return 77;
    }

    // ---- 4. No ext, and an ext with no instruction, agree --------------
    struct transcribe_run_params rp{};
    transcribe_run_params_init(&rp);
    CHECK(transcribe_run(sess, pcm.data(), static_cast<int>(pcm.size()), &rp) == TRANSCRIBE_OK);
    const std::string plain = full_text(sess);
    CHECK(!plain.empty());

    transcribe_run_params_init(&rp);
    rp.family = &ext.ext;  // instruction still NULL
    CHECK(transcribe_run(sess, pcm.data(), static_cast<int>(pcm.size()), &rp) == TRANSCRIBE_OK);
    CHECK(full_text(sess) == plain);

    // ---- 3. Pre-clear rejection preserves the previous result ----------
    // A wrong-kind ext: the generic check rejects it before clear_result.
    struct transcribe_whisper_run_ext foreign{};
    transcribe_whisper_run_ext_init(&foreign);
    transcribe_run_params_init(&rp);
    rp.family = &foreign.ext;
    CHECK(transcribe_run(sess, pcm.data(), static_cast<int>(pcm.size()), &rp) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(full_text(sess) == plain);  // snapshot survived

    // An instruction plus TASK_TRANSLATE: both want the one slot.
    struct transcribe_voxtral_run_ext clash{};
    transcribe_voxtral_run_ext_init(&clash);
    clash.instruction = "Transcribe. Expected terms: NixOS, nixpkgs.";
    transcribe_run_params_init(&rp);
    rp.family          = &clash.ext;
    rp.task            = TRANSCRIBE_TASK_TRANSLATE;
    rp.target_language = "fr";
    CHECK(transcribe_run(sess, pcm.data(), static_cast<int>(pcm.size()), &rp) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(full_text(sess) == plain);  // snapshot survived

    // ---- 5. An instruction is accepted and still transcribes -----------
    struct transcribe_voxtral_run_ext hinted{};
    transcribe_voxtral_run_ext_init(&hinted);
    hinted.instruction = "Transcribe the audio. Expected terms: Americans, country.";
    transcribe_run_params_init(&rp);
    rp.family = &hinted.ext;
    CHECK(transcribe_run(sess, pcm.data(), static_cast<int>(pcm.size()), &rp) == TRANSCRIBE_OK);
    const std::string instructed = full_text(sess);
    CHECK(!instructed.empty());
    std::fprintf(stderr, "plain      : %s\n", plain.c_str());
    std::fprintf(stderr, "instructed : %s\n", instructed.c_str());

    // An empty instruction is the same as none: the family treats "" as
    // absent rather than sending an empty instruct block.
    struct transcribe_voxtral_run_ext empty{};
    transcribe_voxtral_run_ext_init(&empty);
    empty.instruction = "";
    transcribe_run_params_init(&rp);
    rp.family = &empty.ext;
    CHECK(transcribe_run(sess, pcm.data(), static_cast<int>(pcm.size()), &rp) == TRANSCRIBE_OK);
    CHECK(full_text(sess) == plain);

    transcribe_session_free(sess);
    transcribe_model_free(model);

    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::fprintf(stderr, "voxtral run ext: all checks passed\n");
    return 0;
}
