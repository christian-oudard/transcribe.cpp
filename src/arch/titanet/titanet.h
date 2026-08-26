// arch/titanet/titanet.h - TitaNet internal model and session types.
//
// TitaNet is an `encoder-embedding`: five ContextNet blocks of separable
// convolutions with squeeze-excite, attentive statistics pooling, and a
// projection to one 192-dim vector per clip. Two clips of one voice land
// close together under cosine distance. There is no tokenizer, no decoder,
// no text and no timestamps.
//
// It exists to be clustered over. sortformer tracks speakers but caps at
// four and produces no vectors, so a recording with more voices than that
// cannot be told apart at all; embeddings turn "how many speakers" from an
// architectural constant into a property of the audio.
//
// Nothing here is causal or chunkable. Squeeze-excite averages over the
// whole clip and the pooling takes a softmax across it, so every frame
// depends on every other one and a window of the audio is a different
// function rather than an approximation of this one.

#pragma once

#include "transcribe-backend.h"
#include "transcribe-mel.h"
#include "transcribe-model.h"
#include "transcribe-session.h"
#include "weights.h"

#include <optional>
#include <vector>

struct ggml_context;
struct ggml_backend_buffer;
struct ggml_backend_sched;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;
typedef struct ggml_backend_sched *  ggml_backend_sched_t;

namespace transcribe::titanet {

// Family defaults, applied before transcribe::read_capability_kv (KV present
// overrides, KV absent keeps the default). Defined in capabilities.cpp.
void apply_family_invariants(transcribe_model & model);

struct TitanetModel final : public transcribe_model {
    TitanetHParams hparams;
    TitanetWeights weights;

    ggml_context *          ctx_meta = nullptr;
    transcribe::BackendPlan plan;
    ggml_backend_buffer_t   backend_buffer = nullptr;

    // Folded BatchNorm scales and shifts, computed at load from the raw
    // weight / bias / running_mean / running_var tensors.
    ggml_context *        bn_ctx    = nullptr;
    ggml_backend_buffer_t bn_buffer = nullptr;

    std::optional<transcribe::MelFrontend> mel;

    TitanetModel() = default;
    ~TitanetModel() override;

    const transcribe::Tokenizer * tokenizer() const override { return nullptr; }
};

struct TitanetSession final : public transcribe_session {
    ggml_backend_sched_t sched       = nullptr;
    ggml_context *       compute_ctx = nullptr;

    std::vector<float> mel_buf;
    // The product of a run: one vector per clip, kept on the session so a
    // caller that ran the model can read it without a second pass.
    std::vector<float> embedding;

    ~TitanetSession() override;
};

extern const Arch arch;

}  // namespace transcribe::titanet
