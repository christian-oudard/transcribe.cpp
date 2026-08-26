// arch/titanet/model.cpp - TitaNet load, graph and run.
//
// The graph is one shot over the whole clip: mel, five ContextNet blocks,
// attentive statistics pooling, and a projection to the embedding. Working
// layout is [T, C] with time fastest, which is what the conformer conv
// helpers take; the reference dumps the encoder time-major, so the parity
// tensors are transposed on the way out.

#include "titanet.h"
#include "transcribe/titanet.h"

#include "conformer/conformer.h"
#include "ggml.h"
#include "gguf.h"
#include "transcribe-arch.h"
#include "transcribe-debug.h"
#include "transcribe-batch-util.h"
#include "transcribe-load-common.h"
#include "transcribe-loader.h"
#include "transcribe-meta.h"
#include "transcribe-log.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

namespace transcribe::titanet {

namespace conf = transcribe::conformer;

// BatchNorm epsilon, which is not in the GGUF and is not one number. NeMo's
// jasper blocks build theirs with eps=1e-3 (jasper.py, `_get_conv_bn_layer`)
// where everything else takes PyTorch's 1e-5 default. Using 1e-5 throughout
// costs nothing in shape and everything in numbers: block 1's BatchNorm came
// back at cosine 0.54 against the reference, and the encoder diverged from
// there.
static constexpr float kBlockBnEps = 1e-3f;
static constexpr float kBnEps      = 1e-5f;

// AttentivePoolLayer's variance floor. Guards the square root, and only ever
// binds on a channel that is constant across the clip.
static constexpr float kStatEps = 1e-10f;

static constexpr char k_default_variant[] = "speakerverification_en_titanet_large";

TitanetModel::~TitanetModel() {
    if (bn_buffer != nullptr) {
        ggml_backend_buffer_free(bn_buffer);
    }
    if (bn_ctx != nullptr) {
        ggml_free(bn_ctx);
    }
    if (backend_buffer != nullptr) {
        ggml_backend_buffer_free(backend_buffer);
    }
    if (ctx_meta != nullptr) {
        ggml_free(ctx_meta);
    }
}

TitanetSession::~TitanetSession() {
    if (sched != nullptr) {
        ggml_backend_sched_free(sched);
    }
    if (compute_ctx != nullptr) {
        ggml_free(compute_ctx);
    }
}

namespace {

// ---------------------------------------------------------------------------
// BatchNorm folding
// ---------------------------------------------------------------------------

// Every BatchNorm here runs in inference mode, where it is an affine map per
// channel: y = (x - mean)/sqrt(var + eps) * w + b. Folding it at load turns
// eleven tensor reads and a rsqrt per graph into one multiply and one add.
struct BnSlot {
    ggml_tensor *  w;
    ggml_tensor *  b;
    ggml_tensor *  rm;
    ggml_tensor *  rv;
    ggml_tensor ** scale;
    ggml_tensor ** shift;
    float          eps;
};

void collect_bn_slots(TitanetWeights & w, std::vector<BnSlot> & out) {
    for (TitanetBlock & blk : w.blocks) {
        for (TitanetRepeat & rep : blk.reps) {
            out.push_back({ rep.bn_w, rep.bn_b, rep.bn_rm, rep.bn_rv, &rep.scale, &rep.shift, kBlockBnEps });
        }
        if (blk.res_pw != nullptr) {
            out.push_back(
                { blk.res_bn_w, blk.res_bn_b, blk.res_bn_rm, blk.res_bn_rv, &blk.res_scale, &blk.res_shift,
                  kBlockBnEps });
        }
    }
    out.push_back({ w.pool_bn_w, w.pool_bn_b, w.pool_bn_rm, w.pool_bn_rv, &w.pool_bn_scale, &w.pool_bn_shift, kBnEps });
    out.push_back({ w.emb_bn_w, w.emb_bn_b, w.emb_bn_rm, w.emb_bn_rv, &w.emb_bn_scale, &w.emb_bn_shift, kBnEps });
}

transcribe_status fuse_batch_norms(TitanetModel & m) {
    std::vector<BnSlot> slots;
    collect_bn_slots(m.weights, slots);

    const size_t     ctx_size = slots.size() * 2 * ggml_tensor_overhead() + 256;
    ggml_init_params params   = { ctx_size, nullptr, /*no_alloc=*/true };
    m.bn_ctx                  = ggml_init(params);
    if (m.bn_ctx == nullptr) {
        return TRANSCRIBE_ERR_BACKEND;
    }
    for (BnSlot & s : slots) {
        *s.scale = ggml_new_tensor_1d(m.bn_ctx, GGML_TYPE_F32, s.w->ne[0]);
        *s.shift = ggml_new_tensor_1d(m.bn_ctx, GGML_TYPE_F32, s.w->ne[0]);
    }
    m.bn_buffer = ggml_backend_alloc_ctx_tensors(m.bn_ctx, m.plan.scheduler_list.back());
    if (m.bn_buffer == nullptr) {
        return TRANSCRIBE_ERR_BACKEND;
    }

    std::vector<float> bn_w, bn_b, rm, rv, scale, shift;
    for (BnSlot & s : slots) {
        const size_t n     = static_cast<size_t>(s.w->ne[0]);
        const size_t bytes = n * sizeof(float);
        bn_w.resize(n);
        bn_b.resize(n);
        rm.resize(n);
        rv.resize(n);
        scale.resize(n);
        shift.resize(n);
        ggml_backend_tensor_get(s.w, bn_w.data(), 0, bytes);
        ggml_backend_tensor_get(s.b, bn_b.data(), 0, bytes);
        ggml_backend_tensor_get(s.rm, rm.data(), 0, bytes);
        ggml_backend_tensor_get(s.rv, rv.data(), 0, bytes);
        for (size_t c = 0; c < n; ++c) {
            scale[c] = bn_w[c] / std::sqrt(rv[c] + s.eps);
            shift[c] = bn_b[c] - rm[c] * scale[c];
        }
        ggml_backend_tensor_set(*s.scale, scale.data(), 0, bytes);
        ggml_backend_tensor_set(*s.shift, shift.data(), 0, bytes);
    }
    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// Graph pieces. x is [T, C] throughout: time fastest, one channel per row.
// ---------------------------------------------------------------------------

// Pointwise (kernel 1) convolution, which is a per-frame linear map.
ggml_tensor * pointwise(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x, ggml_tensor * bias = nullptr) {
    ggml_tensor * y = conf::conv_1d_f32(ctx, w, x, /*stride=*/1, /*padding=*/0, /*dilation=*/1);
    if (bias != nullptr) {
        y = ggml_add(ctx, y, ggml_reshape_2d(ctx, bias, 1, bias->ne[0]));
    }
    return y;
}

// Mean of every channel over the clip, as [1, C] for broadcasting back.
ggml_tensor * clip_mean(ggml_context * ctx, ggml_tensor * x) {
    return ggml_mean(ctx, x);
}

// Squeeze-excite at se_context_size -1: the context is the whole clip, so the
// squeeze is a plain mean over time and the excitation is one scale per
// channel. Both linears are biasless (jasper.SqueezeExcite).
ggml_tensor * squeeze_excite(ggml_context * ctx, const TitanetBlock & blk, ggml_tensor * x) {
    const int64_t c = x->ne[1];

    ggml_tensor * m = ggml_reshape_1d(ctx, clip_mean(ctx, x), c);
    ggml_tensor * y = ggml_mul_mat(ctx, blk.se_down, m);
    y               = ggml_relu(ctx, y);
    y               = ggml_mul_mat(ctx, blk.se_up, y);
    y               = ggml_sigmoid(ctx, y);
    return ggml_mul(ctx, x, ggml_reshape_2d(ctx, y, 1, c));
}

// One ContextNet block: `repeat` separable convolutions, squeeze-excite, the
// skip path where the block declares one, and the activation last.
//
// The order matters and is not the obvious one. NeMo puts the final repeat's
// ReLU after the residual add rather than before it (JasperBlock keeps it in
// `mout`), so a port that activates every repeat uniformly is wrong in a way
// that leaves every shape intact.
ggml_tensor * block(ggml_context * ctx, const TitanetBlock & blk, ggml_tensor * x, int kernel) {
    ggml_tensor * in  = x;
    ggml_tensor * h   = x;
    const int     pad = (kernel - 1) / 2;

    for (size_t r = 0; r < blk.reps.size(); ++r) {
        const TitanetRepeat & rep = blk.reps[r];
        h                         = conf::conv_1d_dw_f32(ctx, rep.dw, h, /*s=*/1, /*p=*/pad, /*d=*/1);
        h                         = pointwise(ctx, rep.pw, h);
        h                         = conf::fused_batch_norm(ctx, h, rep.scale, rep.shift);
        if (r + 1 < blk.reps.size()) {
            h = ggml_relu(ctx, h);
        }
    }
    h = squeeze_excite(ctx, blk, h);

    if (blk.res_pw != nullptr) {
        ggml_tensor * res = pointwise(ctx, blk.res_pw, in);
        res               = conf::fused_batch_norm(ctx, res, blk.res_scale, blk.res_shift);
        h                 = ggml_add(ctx, h, res);
    }
    return ggml_relu(ctx, h);
}

// Weighted mean and standard deviation of every channel over the clip, given
// per-frame weights that sum to one. This is get_statistics_with_mask: the
// deviation is squared against the weighted mean and the sum is floored
// before the square root.
struct Stats {
    ggml_tensor * mean;  // [1, C]
    ggml_tensor * dev;   // [1, C]
};

Stats weighted_stats(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w) {
    ggml_tensor * mean = ggml_sum_rows(ctx, ggml_mul(ctx, x, w));
    ggml_tensor * d    = ggml_sub(ctx, x, ggml_repeat(ctx, mean, x));
    ggml_tensor * var  = ggml_sum_rows(ctx, ggml_mul(ctx, ggml_sqr(ctx, d), w));
    return { mean, ggml_sqrt(ctx, ggml_clamp(ctx, var, kStatEps, INFINITY)) };
}

}  // namespace

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------

// One forward pass over a clip of T mel frames. Built once and computed many
// times: diarization runs this same shape over every window of a recording,
// where rebuilding the graph per window would cost more than the graph.
EmbedGraph build_embed_graph(ggml_context * ctx, const TitanetModel & m, int T, int n_mels) {
    EmbedGraph g;
    g.graph = ggml_new_graph_custom(ctx, /*size=*/8192, /*grads=*/false);

    g.mel_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, n_mels);
    ggml_set_input(g.mel_in);

    const TitanetHParams & hp = m.hparams;
    ggml_tensor *          x  = g.mel_in;

        for (int b = 0; b < hp.enc_n_blocks; ++b) {
        x = block(ctx, m.weights.blocks[static_cast<size_t>(b)], x, hp.kernel[static_cast<size_t>(b)]);
        g.blocks.push_back(x);
    }
    ggml_tensor * enc_out = x;

    // Attentive statistics pooling. The attention input is the encoder output
    // beside its own clip mean and deviation, so the weights a channel gets
    // depend on how that channel behaved over the whole clip.
    const int64_t c        = enc_out->ne[1];
    g.uniform = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, 1);
    ggml_set_input(g.uniform);
    Stats         clip     = weighted_stats(ctx, enc_out, g.uniform);
    ggml_tensor * attn_in  = ggml_concat(ctx, enc_out, ggml_repeat(ctx, clip.mean, enc_out), 1);
    attn_in                = ggml_concat(ctx, attn_in, ggml_repeat(ctx, clip.dev, enc_out), 1);

    ggml_tensor * a = pointwise(ctx, m.weights.pool_attn0_w, attn_in, m.weights.pool_attn0_b);
    a               = ggml_relu(ctx, a);
    a               = conf::fused_batch_norm(ctx, a, m.weights.pool_bn_scale, m.weights.pool_bn_shift);
    a               = ggml_tanh(ctx, a);
    a               = pointwise(ctx, m.weights.pool_attn1_w, a, m.weights.pool_attn1_b);
    // Softmax over time, one distribution per channel; ne0 is time.
    ggml_tensor * alpha = ggml_soft_max(ctx, a);

    Stats         pooled   = weighted_stats(ctx, enc_out, alpha);
    g.pool_out = ggml_reshape_1d(ctx, ggml_concat(ctx, pooled.mean, pooled.dev, 1), 2 * c);

    ggml_tensor * e = ggml_add(ctx, ggml_mul(ctx, g.pool_out, m.weights.emb_bn_scale), m.weights.emb_bn_shift);
    e               = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, m.weights.emb_proj_w, 2 * c, hp.embedding_size), e);
    g.emb = ggml_add(ctx, e, m.weights.emb_proj_b);

    ggml_build_forward_expand(g.graph, g.emb);
    return g;
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

namespace {

transcribe_status load(Loader & loader, const transcribe_model_load_params * params, transcribe_model ** out_model) {
    const int64_t t_load_start = ggml_time_us();

    auto m       = std::make_unique<TitanetModel>();
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
    if (const transcribe_status st = read_titanet_hparams(loader.gguf(), m->hparams); st != TRANSCRIBE_OK) {
        return st;
    }

    {
        transcribe::MelConfig cfg{};
        cfg.sample_rate  = m->hparams.fe_sample_rate;
        cfg.num_mels     = m->hparams.fe_num_mels;
        cfg.n_fft        = m->hparams.fe_n_fft;
        cfg.win_length   = m->hparams.fe_win_length;
        cfg.hop_length   = m->hparams.fe_hop_length;
        cfg.pre_emphasis = m->hparams.fe_pre_emphasis;
        cfg.normalize    = m->hparams.fe_normalize;
        cfg.pad_mode     = "constant";
        // NeMo's ceil(n/hop) framing, as everywhere else NeMo is the source.
        cfg.nemo_seq_len_ceil = true;
        m->mel.emplace(cfg);
    }

    gguf_init_params init_params{};
    init_params.no_alloc     = true;
    init_params.ctx          = &m->ctx_meta;
    gguf_context * gguf_data = gguf_init_from_file(loader.path().c_str(), init_params);
    if (gguf_data == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    if (const transcribe_status st = build_titanet_weights(m->ctx_meta, m->hparams, m->weights);
        st != TRANSCRIBE_OK) {
        gguf_free(gguf_data);
        return st;
    }

    const transcribe_backend_request backend_req = (params != nullptr) ? params->backend : TRANSCRIBE_BACKEND_AUTO;
    if (const transcribe_status st = transcribe::load_common::init_backends(
            backend_req, (params != nullptr) ? params->gpu_device : 0, "titanet", m->plan);
        st != TRANSCRIBE_OK) {
        gguf_free(gguf_data);
        return st;
    }
    m->backend         = ggml_backend_name(m->plan.primary);
    m->primary_backend = m->plan.primary;

    ggml_backend_buffer_t weights_buffer = ggml_backend_alloc_ctx_tensors(m->ctx_meta, m->plan.primary);
    if (weights_buffer == nullptr) {
        gguf_free(gguf_data);
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "titanet: ggml_backend_alloc_ctx_tensors failed");
        return TRANSCRIBE_ERR_GGUF;
    }
    m->backend_buffer = weights_buffer;
    ggml_backend_buffer_set_usage(weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    if (const transcribe_status st =
            transcribe::load_common::stream_tensor_data(loader.path(), gguf_data, m->ctx_meta, "titanet");
        st != TRANSCRIBE_OK) {
        gguf_free(gguf_data);
        return st;
    }
    gguf_free(gguf_data);

    if (const transcribe_status st = fuse_batch_norms(*m); st != TRANSCRIBE_OK) {
        return st;
    }

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
    auto pc       = std::make_unique<TitanetSession>();
    pc->model     = model;
    pc->n_threads = params->n_threads;
    pc->kv_type   = params->kv_type;
    *out_ctx      = pc.release();
    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

// The reference dumps the encoder time-major, [T, C] slow-to-fast, where the
// graph carries the transpose of that. Only parity dumps care.
ggml_tensor * time_major(ggml_context * ctx, ggml_tensor * x) {
    return ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
}

transcribe_status run(transcribe_session *          session,
                      const float *                 pcm,
                      int                           n_samples,
                      const transcribe_run_params * params) {
    auto * pc = static_cast<TitanetSession *>(session);
    auto * pm = static_cast<TitanetModel *>(session->model);

    if (pc->poll_abort()) {
        return TRANSCRIBE_ERR_ABORTED;
    }
    pc->clear_result();
    pc->embedding.clear();
    transcribe::debug::init();

    // Diarizing is the product. Embedding one clip is what the parity gate and
    // the verification pair need, and it is what a caller gets by not asking
    // for diarization.
    if (params != nullptr && params->diarize == TRANSCRIBE_DIARIZE_MODE_ON) {
        int32_t                                 num_speakers = 0;
        float                                   threshold    = 0.0f;
        std::vector<transcribe::diarize::Region> speech;
        if (params->family != nullptr) {
            const auto * ext = reinterpret_cast<const transcribe_titanet_diarize_ext *>(params->family);
            num_speakers     = ext->num_speakers;
            threshold        = ext->threshold;
            for (int32_t i = 0; i < ext->n_speech; ++i) {
                speech.push_back({ ext->speech_ms[2 * i], ext->speech_ms[2 * i + 1] });
            }
        }
        if (const transcribe_status st = diarize(pc, pm, pcm, n_samples, num_speakers, threshold, speech);
            st != TRANSCRIBE_OK) {
            return st;
        }
        pc->result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
        pc->has_result  = true;
        return TRANSCRIBE_OK;
    }

    if (!pm->mel.has_value()) {
        return TRANSCRIBE_ERR_GGUF;
    }
    int mel_n_mels = 0, T = 0;
    if (const transcribe_status st = pm->mel->compute(pcm, static_cast<size_t>(n_samples), pc->mel_buf, mel_n_mels, T,
                                                      pc->n_threads);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (T <= 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "titanet: %d samples is too short to embed", n_samples);
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    if (pc->compute_ctx != nullptr) {
        ggml_free(pc->compute_ctx);
        pc->compute_ctx = nullptr;
    }
    ggml_init_params ip{};
    ip.mem_size     = 16 * 1024 * 1024;
    ip.mem_buffer   = nullptr;
    ip.no_alloc     = true;
    pc->compute_ctx = ggml_init(ip);
    if (pc->compute_ctx == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }
    ggml_context * ctx = pc->compute_ctx;
    EmbedGraph     g   = build_embed_graph(ctx, *pm, T, mel_n_mels);

    ggml_cgraph *              graph     = g.graph;
    ggml_tensor *              mel_in    = g.mel_in;
    ggml_tensor *              uniform   = g.uniform;
    ggml_tensor *              pool_out  = g.pool_out;
    ggml_tensor *              emb       = g.emb;
    std::vector<ggml_tensor *> block_out = g.blocks;

    std::vector<ggml_tensor *> dumps;
    if (transcribe::debug::enabled()) {
        for (ggml_tensor * b : block_out) {
            dumps.push_back(time_major(ctx, b));
        }
        for (ggml_tensor * d : dumps) {
            ggml_build_forward_expand(graph, d);
            transcribe::debug::mark_tensor_for_dump(d);
        }
        transcribe::debug::mark_tensor_for_dump(pool_out);
        transcribe::debug::mark_tensor_for_dump(emb);
    }

    if (pc->sched == nullptr) {
        pc->sched = ggml_backend_sched_new(pm->plan.scheduler_list.data(), nullptr,
                                           static_cast<int>(pm->plan.scheduler_list.size()),
                                           /*graph_size=*/8192, /*parallel=*/false, /*op_offload=*/true);
        if (pc->sched == nullptr) {
            return TRANSCRIBE_ERR_BACKEND;
        }
    }
    ggml_backend_sched_reset(pc->sched);
    if (!ggml_backend_sched_alloc_graph(pc->sched, graph)) {
        return TRANSCRIBE_ERR_BACKEND;
    }

    ggml_backend_tensor_set(mel_in, pc->mel_buf.data(), 0, pc->mel_buf.size() * sizeof(float));
    // The unweighted statistics are the weighted ones with every frame worth
    // the same, which keeps one code path for both.
    std::vector<float> even(static_cast<size_t>(T), 1.0f / static_cast<float>(T));
    ggml_backend_tensor_set(uniform, even.data(), 0, even.size() * sizeof(float));
    transcribe::debug::dump_tensor("enc.mel.in", mel_in, "frontend");

    transcribe::configure_sched_n_threads(pc->sched, pc->n_threads);
    if (ggml_backend_sched_graph_compute(pc->sched, graph) != GGML_STATUS_SUCCESS) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "titanet: graph_compute failed");
        return TRANSCRIBE_ERR_BACKEND;
    }

    if (transcribe::debug::enabled()) {
        char name[32];
        for (size_t i = 0; i < dumps.size(); ++i) {
            std::snprintf(name, sizeof(name), "enc.block.%zu.out", i);
            transcribe::debug::dump_tensor(name, dumps[i], "encoder");
        }
        transcribe::debug::dump_tensor("enc.out", dumps.back(), "encoder");
        transcribe::debug::dump_tensor("pool.out", pool_out, "pooling");
        transcribe::debug::dump_tensor("emb.out", emb, "embedding");
    }

    pc->embedding.resize(static_cast<size_t>(pm->hparams.embedding_size));
    ggml_backend_tensor_get(emb, pc->embedding.data(), 0, pc->embedding.size() * sizeof(float));

    pc->result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
    pc->has_result  = true;
    return TRANSCRIBE_OK;
}

}  // namespace

namespace {

// The count-or-threshold extension is the only surface here, and only on the
// run slot: there is no push-audio entry point for a model that has to see a
// whole recording before it can cluster it.
bool accepts_ext_kind(const transcribe_model * model, transcribe_ext_slot slot, uint32_t kind) {
    if (model == nullptr || slot != TRANSCRIBE_EXT_SLOT_RUN) {
        return false;
    }
    return kind == TRANSCRIBE_EXT_KIND_TITANET_DIARIZE;
}

// Rejected before the previous result is cleared, so a typo costs nothing.
transcribe_status run_validate(const transcribe_session * /*ctx*/, const transcribe_run_params * params) {
    if (params == nullptr || params->family == nullptr) {
        return TRANSCRIBE_OK;
    }
    if (const transcribe_status st = transcribe_ext_check(params->family, TRANSCRIBE_EXT_KIND_TITANET_DIARIZE,
                                                          sizeof(struct transcribe_titanet_diarize_ext));
        st != TRANSCRIBE_OK) {
        return st;
    }
    const auto * ext = reinterpret_cast<const transcribe_titanet_diarize_ext *>(params->family);
    if (ext->num_speakers < 0 || ext->threshold < 0.0f || ext->threshold > 2.0f) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    // A count with no array, or an array with no count, is a caller that has
    // half-filled the struct; running on that would silently ignore the speech
    // they meant to supply.
    if ((ext->n_speech > 0) != (ext->speech_ms != nullptr)) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (ext->n_speech < 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    return TRANSCRIBE_OK;
}

}  // namespace

extern const Arch arch = {
    /* .name             = */ "titanet",
    /* .load             = */ load,
    /* .init_context     = */ init_context,
    /* .run              = */ run,
    /* .run_batch        = */ nullptr,
    /* .stream_validate  = */ nullptr,
    /* .stream_begin     = */ nullptr,
    /* .stream_feed      = */ nullptr,
    /* .stream_finalize  = */ nullptr,
    /* .stream_reset     = */ nullptr,
    /* .accepts_ext_kind = */ accepts_ext_kind,
    /* .run_validate     = */ run_validate,
};

// Public extension init (global scope, C linkage), kept here so
// transcribe.cpp stays family-agnostic.
extern "C" void transcribe_titanet_diarize_ext_init(struct transcribe_titanet_diarize_ext * p) {
    if (p == nullptr) {
        return;
    }
    std::memset(p, 0, sizeof(*p));
    p->ext.size = sizeof(*p);
    p->ext.kind = TRANSCRIBE_EXT_KIND_TITANET_DIARIZE;
}

}  // namespace transcribe::titanet
