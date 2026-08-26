// arch/fsmn_vad/model.cpp - FSMN VAD load, graph and run.
//
// Working layout is [C, T]: one column per frame, which is what the linear
// layers want, since ggml_mul_mat contracts the fastest axis. The memory
// convolution is the exception and runs over time, so the tensor is permuted
// around it and back.

#include "fsmn_vad.h"

#include "diarize/regions.h"

#include "conformer/conformer.h"
#include "ggml.h"
#include "gguf.h"
#include "transcribe-arch.h"
#include "transcribe-batch-util.h"
#include "transcribe-debug.h"
#include "transcribe-load-common.h"
#include "transcribe-loader.h"
#include "transcribe-log.h"
#include "transcribe-meta.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

namespace transcribe::fsmn_vad {

namespace conf = transcribe::conformer;
namespace dz   = transcribe::diarize;

static constexpr char k_default_variant[] = "fsmn-vad";

// A frame is speech when the model says so with more confidence than not.
// Halfway is where FunASR's own decision sits before its smoothing, and the
// smoothing below is where the real policy lives.
static constexpr float kSpeechThreshold = 0.5f;

// Gaps shorter than this do not end a region, and regions shorter than this
// are not regions. Both are what turn a per-frame probability into something
// worth calling a stretch of speech: a breath inside a sentence is not
// silence, and a single frame above the threshold is not somebody talking.
static constexpr int64_t kMinSilenceMs = 300;
static constexpr int64_t kMinSpeechMs  = 100;

// Frames per graph. The whole recording in one graph is the obvious
// implementation and it asks a card for eight gigabytes on a two-hour file --
// the activations are a few hundred floats per frame and there are six hundred
// thousand frames.
//
// Chunking is exact here rather than an approximation, which is the reason
// this model was chosen: the memory is causal and reaches exactly lorder-1
// frames back, so a chunk that carries that many frames of context computes
// the same numbers the whole-file graph would. Five minutes is small enough
// for any card that can hold the weights and large enough that the context is
// a rounding error.
static constexpr int kChunkFrames = 30000;

FsmnVadModel::~FsmnVadModel() {
    if (backend_buffer != nullptr) {
        transcribe::safe_buffer_free(backend_buffer);
    }
    if (ctx_meta != nullptr) {
        ggml_free(ctx_meta);
    }
}

FsmnVadSession::~FsmnVadSession() {
    if (sched != nullptr) {
        transcribe::safe_sched_free(sched);
    }
    if (compute_ctx != nullptr) {
        ggml_free(compute_ctx);
    }
}

namespace {

// y = W^T x + b, with x as [in, T] and the bias broadcast across the frames.
ggml_tensor * affine(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x, ggml_tensor * b) {
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    if (b != nullptr) {
        y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, b->ne[0], 1));
    }
    return y;
}

// The memory: each channel sees the last `lorder` frames of itself, weighted,
// added back to the present. Left only, so a frame's answer never depends on
// audio after it -- which is what makes this usable on a stream and what makes
// a file and a stream agree.
ggml_tensor * memory(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * x, int lorder) {
    // The two published files disagree about which axis of the kernel is
    // time. The convolution below wants time fastest, so a file that stores
    // [C, lorder] is turned round here: four kernels of 2560 floats, once per
    // graph build.
    if (weight->ne[0] != lorder) {
        weight = ggml_cont(ctx, ggml_transpose(ctx, weight));
    }
    // [C, T] -> [T, C] for the convolution, which runs over the fastest axis.
    ggml_tensor * time_major = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    ggml_tensor * kernel     = ggml_reshape_3d(ctx, weight, weight->ne[0], 1, weight->ne[1]);
    // Padding the left by lorder-1 and none on the right is the causal half of
    // a same-length convolution.
    ggml_tensor * y = conf::conv_1d_dw_f32(ctx, kernel, time_major, /*s=*/1, /*p=*/lorder - 1, /*d=*/1);
    // conv_1d_dw_f32 pads both sides, so the tail carries frames the model
    // must not see: keep the first T outputs, which are the causal ones.
    y = ggml_cont(ctx, ggml_view_2d(ctx, y, time_major->ne[0], y->ne[1], y->nb[1], 0));
    y = ggml_cont(ctx, ggml_permute(ctx, y, 1, 0, 2, 3));
    return ggml_add(ctx, x, y);
}

struct Graph {
    ggml_cgraph * graph = nullptr;
    ggml_tensor * in    = nullptr;  // [input_dim, T]
    ggml_tensor * probs = nullptr;  // [output_dim, T]
};

Graph build(ggml_context * ctx, const FsmnVadModel & m, int frames) {
    const FsmnVadHParams & hp = m.hparams;
    Graph                  g;
    g.graph = ggml_new_graph(ctx);
    g.in    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.input_dim, frames);
    ggml_set_input(g.in);

    ggml_tensor * x = affine(ctx, m.weights.in0_w, g.in, m.weights.in0_b);
    x               = affine(ctx, m.weights.in1_w, x, m.weights.in1_b);
    x               = ggml_relu(ctx, x);

    for (const FsmnLayer & layer : m.weights.layers) {
        ggml_tensor * h = ggml_mul_mat(ctx, layer.linear, x);  // no bias, by design
        h               = memory(ctx, layer.memory, h, hp.lorder);
        h               = affine(ctx, layer.affine_w, h, layer.affine_b);
        x               = ggml_relu(ctx, h);
    }

    x       = affine(ctx, m.weights.out0_w, x, m.weights.out0_b);
    x       = affine(ctx, m.weights.out1_w, x, m.weights.out1_b);
    g.probs = ggml_soft_max(ctx, x);  // over the states, which are ne0
    ggml_build_forward_expand(g.graph, g.probs);
    return g;
}

// Put the detector's regions on the session, as speaker rows with one
// anonymous speaker: somebody was talking, and this model does not know who.
void publish(FsmnVadSession * pc, const std::vector<float> & speech, double frame_ms) {
    dz::RegionRules rules;
    rules.frame_ms      = frame_ms;
    rules.threshold     = kSpeechThreshold;
    rules.min_silence_ms = kMinSilenceMs;
    rules.min_speech_ms  = kMinSpeechMs;
    for (const dz::Region & r : dz::regions_from_frames(speech.data(), static_cast<int32_t>(speech.size()), rules)) {
        transcribe_session::SpeakerSegmentEntry row;
        row.t0_ms      = r.t0_ms;
        row.t1_ms      = r.t1_ms;
        row.speaker_id = 1;
        row.p          = std::numeric_limits<float>::quiet_NaN();
        pc->speaker_segments.push_back(row);
    }
}

transcribe_status load(Loader & loader, const transcribe_model_load_params * params, transcribe_model ** out_model) {
    const int64_t t_load_start = ggml_time_us();

    auto m       = std::make_unique<FsmnVadModel>();
    m->arch      = &arch;
    m->t_load_us = 0;
    m->variant   = loader.variant().empty() ? k_default_variant : loader.variant();
    m->backend.clear();

    apply_family_invariants(*m);
    m->caps.n_languages = 0;
    m->caps.languages   = nullptr;
    if (const transcribe_status st = read_capability_kv(loader.gguf(), m->caps); st != TRANSCRIBE_OK) {
        return st;
    }
    if (const transcribe_status st = read_languages_kv(loader.gguf(), *m); st != TRANSCRIBE_OK) {
        return st;
    }
    if (const transcribe_status st = read_fsmn_vad_hparams(loader.gguf(), m->hparams); st != TRANSCRIBE_OK) {
        return st;
    }

    gguf_init_params init_params{};
    init_params.no_alloc     = true;
    init_params.ctx          = &m->ctx_meta;
    gguf_context * gguf_data = gguf_init_from_file(loader.path().c_str(), init_params);
    if (gguf_data == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }
    if (const transcribe_status st = build_fsmn_vad_weights(m->ctx_meta, m->hparams, m->weights);
        st != TRANSCRIBE_OK) {
        gguf_free(gguf_data);
        return st;
    }

    const transcribe_backend_request backend_req = (params != nullptr) ? params->backend : TRANSCRIBE_BACKEND_AUTO;
    if (const transcribe_status st = transcribe::load_common::init_backends(
            backend_req, (params != nullptr) ? params->gpu_device : 0, "fsmn-vad", m->plan);
        st != TRANSCRIBE_OK) {
        gguf_free(gguf_data);
        return st;
    }
    m->backend         = ggml_backend_name(m->plan.primary);
    m->primary_backend = m->plan.primary;

    m->backend_buffer = ggml_backend_alloc_ctx_tensors(m->ctx_meta, m->plan.primary);
    if (m->backend_buffer == nullptr) {
        gguf_free(gguf_data);
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "fsmn-vad: ggml_backend_alloc_ctx_tensors failed");
        return TRANSCRIBE_ERR_GGUF;
    }
    ggml_backend_buffer_set_usage(m->backend_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    if (const transcribe_status st =
            transcribe::load_common::stream_tensor_data(loader.path(), gguf_data, m->ctx_meta, "fsmn-vad");
        st != TRANSCRIBE_OK) {
        gguf_free(gguf_data);
        return st;
    }
    gguf_free(gguf_data);

    // The frontend needs the normalization vectors as host memory, so they are
    // read back off the device once here rather than per run.
    transcribe::KaldiFbankParams fe{};
    fe.n_mels      = m->hparams.fe_num_mels;
    fe.sample_rate = m->hparams.fe_sample_rate;
    fe.win_length  = m->hparams.fe_win_length;
    fe.hop_length  = m->hparams.fe_hop_length;
    fe.lfr_m       = m->hparams.fe_lfr_m;
    fe.lfr_n       = m->hparams.fe_lfr_n;
    fe.d_input     = m->hparams.input_dim;
    fe.apply_cmvn  = true;
    fe.cmvn_shift.resize(static_cast<size_t>(m->hparams.input_dim));
    fe.cmvn_scale.resize(static_cast<size_t>(m->hparams.input_dim));
    const size_t cmvn_bytes = static_cast<size_t>(m->hparams.input_dim) * sizeof(float);
    ggml_backend_tensor_get(m->weights.cmvn_shift, fe.cmvn_shift.data(), 0, cmvn_bytes);
    ggml_backend_tensor_get(m->weights.cmvn_scale, fe.cmvn_scale.data(), 0, cmvn_bytes);
    m->frontend = std::make_unique<transcribe::KaldiFbankFrontend>(fe);

    m->t_load_us = ggml_time_us() - t_load_start;
    *out_model   = m.release();
    return TRANSCRIBE_OK;
}

transcribe_status init_context(transcribe_model *                model,
                               const transcribe_session_params * params,
                               transcribe_session **             out_ctx) {
    if (model->arch != &arch) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    auto pc       = std::make_unique<FsmnVadSession>();
    pc->model     = model;
    pc->n_threads = params->n_threads;
    pc->kv_type   = params->kv_type;
    *out_ctx      = pc.release();
    return TRANSCRIBE_OK;
}

transcribe_status run(transcribe_session *          session,
                      const float *                 pcm,
                      int                           n_samples,
                      const transcribe_run_params * /*params*/) {
    auto * pc = static_cast<FsmnVadSession *>(session);
    auto * pm = static_cast<FsmnVadModel *>(session->model);

    if (pc->poll_abort()) {
        return TRANSCRIBE_ERR_ABORTED;
    }
    pc->clear_result();
    pc->speech.clear();
    transcribe::debug::init();

    const int frames = pm->frontend->compute(pcm, static_cast<size_t>(n_samples), pc->features);
    if (frames <= 0) {
        // Too short to make a single frame. No speech is a legitimate answer.
        pc->result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
        pc->has_result  = true;
        return TRANSCRIBE_OK;
    }

    if (pc->compute_ctx != nullptr) {
        ggml_free(pc->compute_ctx);
    }
    ggml_init_params ip{};
    // One context for every chunk's graph: the tensors are small and the
    // scheduler owns the buffers, so the arena only holds descriptions.
    ip.mem_size     = 64 * 1024 * 1024;
    ip.no_alloc     = true;
    pc->compute_ctx = ggml_init(ip);
    if (pc->compute_ctx == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    if (pc->sched == nullptr) {
        pc->sched = ggml_backend_sched_new(pm->plan.scheduler_list.data(), nullptr,
                                           static_cast<int>(pm->plan.scheduler_list.size()),
                                           /*graph_size=*/2048, /*parallel=*/false, /*op_offload=*/true);
        if (pc->sched == nullptr) {
            return TRANSCRIBE_ERR_BACKEND;
        }
    }
    transcribe::configure_sched_n_threads(pc->sched, pc->n_threads);

    const int          dim     = pm->hparams.input_dim;
    const int          states  = pm->hparams.output_dim;
    const int          context = pm->hparams.lorder - 1;
    std::vector<float> posteriors;
    pc->speech.assign(static_cast<size_t>(frames), 0.0f);

    for (int at = 0; at < frames; at += kChunkFrames) {
        if (pc->poll_abort()) {
            return TRANSCRIBE_ERR_ABORTED;
        }
        // Everything but the first chunk starts inside the one before it, far
        // enough back that the memory sees what it would have seen; those
        // frames are computed and thrown away.
        const int from  = (at > context) ? at - context : 0;
        const int to    = std::min(at + kChunkFrames, frames);
        const int width = to - from;

        Graph g = build(pc->compute_ctx, *pm, width);
        ggml_backend_sched_reset(pc->sched);
        if (!ggml_backend_sched_alloc_graph(pc->sched, g.graph)) {
            return TRANSCRIBE_ERR_BACKEND;
        }
        ggml_backend_tensor_set(g.in, pc->features.data() + static_cast<size_t>(from) * dim, 0,
                                static_cast<size_t>(width) * dim * sizeof(float));
        if (from == 0) {
            transcribe::debug::dump_tensor("enc.feats.in", g.in, "frontend");
        }
        if (ggml_backend_sched_graph_compute(pc->sched, g.graph) != GGML_STATUS_SUCCESS) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "fsmn-vad: graph_compute failed");
            return TRANSCRIBE_ERR_BACKEND;
        }
        if (from == 0) {
            transcribe::debug::dump_tensor("vad.posteriors", g.probs, "head");
        }

        posteriors.resize(static_cast<size_t>(width) * states);
        ggml_backend_tensor_get(g.probs, posteriors.data(), 0, posteriors.size() * sizeof(float));
        for (int t = at; t < to; ++t) {
            const size_t row = static_cast<size_t>(t - from) * states + static_cast<size_t>(pm->hparams.silence_pdf);
            pc->speech[static_cast<size_t>(t)] = 1.0f - posteriors[row];
        }
    }
    if (transcribe::debug::enabled()) {
        const long long shape[] = { frames };
        transcribe::debug::dump_host_f32("vad.speech", pc->speech.data(), frames, shape, 1, "head");
    }

    publish(pc, pc->speech, pm->hparams.frame_ms());
    pc->result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
    pc->has_result  = true;
    return TRANSCRIBE_OK;
}

}  // namespace

extern const Arch arch = {
    /* .name             = */ "fsmn-vad",
    /* .load             = */ load,
    /* .init_context     = */ init_context,
    /* .run              = */ run,
    /* .run_batch        = */ nullptr,
    /* .stream_validate  = */ nullptr,
    /* .stream_begin     = */ nullptr,
    /* .stream_feed      = */ nullptr,
    /* .stream_finalize  = */ nullptr,
    /* .stream_reset     = */ nullptr,
    /* .accepts_ext_kind = */ nullptr,
    /* .run_validate     = */ nullptr,
};

}  // namespace transcribe::fsmn_vad
