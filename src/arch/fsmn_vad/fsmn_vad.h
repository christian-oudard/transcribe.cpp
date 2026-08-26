// arch/fsmn_vad/fsmn_vad.h - FSMN voice activity detector.
//
// Answers one question per frame: is anybody speaking. Not who, and not what.
//
// It is here because diarization needs it. Clustering speaker embeddings
// counts an extra speaker on most meetings, and the extra cluster is audible
// non-speech -- a chair, a keyboard -- which an energy threshold cannot tell
// from a voice and which then groups coherently enough to look like a quiet
// participant. Five gates that are not models were measured first and none
// worked; see docs/diarization.md.
//
// The output is speech regions, published on the speaker-segment surface with
// one anonymous speaker. That is a deliberate reading of "who spoke when":
// somebody did, and this model does not know or claim who. A caller wanting
// names runs a diarizer over these regions.

#pragma once

#include "transcribe-backend.h"
#include "transcribe-kaldi-fbank.h"
#include "transcribe-model.h"
#include "transcribe-session.h"
#include "weights.h"

#include <memory>
#include <vector>

struct ggml_context;
struct ggml_backend_buffer;
struct ggml_backend_sched;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;
typedef struct ggml_backend_sched *  ggml_backend_sched_t;

namespace transcribe::fsmn_vad {

void apply_family_invariants(transcribe_model & model);

struct FsmnVadModel final : public transcribe_model {
    FsmnVadHParams hparams;
    FsmnVadWeights weights;

    ggml_context *          ctx_meta = nullptr;
    transcribe::BackendPlan plan;
    ggml_backend_buffer_t   backend_buffer = nullptr;

    // The frontend is kaldi fbank with LFR stacking and CMVN, which the other
    // FunASR models here already use. It is built once at load, since the
    // window and the filterbank do not change.
    std::unique_ptr<transcribe::KaldiFbankFrontend> frontend;

    FsmnVadModel() = default;
    ~FsmnVadModel() override;

    const transcribe::Tokenizer * tokenizer() const override { return nullptr; }
};

struct FsmnVadSession final : public transcribe_session {
    ggml_backend_sched_t sched       = nullptr;
    ggml_context *       compute_ctx = nullptr;

    std::vector<float> features;
    // One speech probability per frame, kept so a caller can read the curve
    // rather than only the regions cut from it.
    std::vector<float> speech;

    ~FsmnVadSession() override;
};

extern const Arch arch;

}  // namespace transcribe::fsmn_vad
