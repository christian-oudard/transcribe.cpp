// arch/titanet/weights.cpp - TitaNet hparam KV reader and weight catalog.

#include "weights.h"

#include "ggml.h"
#include "gguf.h"
#include "transcribe-log.h"

#include <cstdio>
#include <string>
#include <vector>

namespace transcribe::titanet {

namespace {

transcribe_status kv_u32(const gguf_context * g, const char * key, int32_t & out) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "titanet: missing KV %s", key);
        return TRANSCRIBE_ERR_GGUF;
    }
    out = static_cast<int32_t>(gguf_get_val_u32(g, id));
    return TRANSCRIBE_OK;
}

transcribe_status kv_f32(const gguf_context * g, const char * key, float & out) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "titanet: missing KV %s", key);
        return TRANSCRIBE_ERR_GGUF;
    }
    out = gguf_get_val_f32(g, id);
    return TRANSCRIBE_OK;
}

transcribe_status kv_str(const gguf_context * g, const char * key, std::string & out) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "titanet: missing KV %s", key);
        return TRANSCRIBE_ERR_GGUF;
    }
    out = gguf_get_val_str(g, id);
    return TRANSCRIBE_OK;
}

// Per-block integer array: filters, repeat, kernel and the rest arrive as one
// KV array each, so a block's shape is read across five of them rather than
// from a per-block namespace.
transcribe_status kv_i32_array(const gguf_context * g, const char * key, int32_t n, std::vector<int32_t> & out) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "titanet: missing KV %s", key);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (gguf_get_arr_n(g, id) != static_cast<size_t>(n)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "titanet: KV %s has %zu entries, want %d", key, gguf_get_arr_n(g, id), n);
        return TRANSCRIBE_ERR_GGUF;
    }
    const auto * data = static_cast<const int32_t *>(gguf_get_arr_data(g, id));
    out.assign(data, data + n);
    return TRANSCRIBE_OK;
}

}  // namespace

// A second GGUF of this checkpoint is published, and it describes the model in
// its own vocabulary: `titanet.channels` and `titanet.block_kernels` where
// this tree writes `stt.titanet.encoder.filters` and `.kernel`, and no
// frontend block at all -- it bakes the mel filterbank and the window in as
// tensors instead. The weights are the same to F16 rounding, checked tensor by
// tensor, so what is missing here is vocabulary rather than information.
//
// What it does not say is filled from the architecture: every stride and
// dilation in TitaNet is 1, the activation is ReLU and the pooling is
// attentive statistics, and the frontend is NeMo's standard 80-mel
// preprocessor. Which blocks carry a residual is not guessed -- it is read off
// the tensors, since a block with a skip path has the convolution for it.
transcribe_status read_published_hparams(const gguf_context * g, TitanetHParams & hp) {
    int32_t channels = 0, epilog = 0;
#define RD(key, out)                                                        \
    if (const transcribe_status st = kv_u32(g, key, out); st != TRANSCRIBE_OK) \
    return st
    RD("titanet.emb_dim", hp.embedding_size);
    RD("titanet.n_mels", hp.enc_feat_in);
    RD("titanet.n_blocks", hp.enc_n_blocks);
    RD("titanet.channels", channels);
    RD("titanet.epilog_channels", epilog);
#undef RD
    hp.enc_feat_out = epilog;

    const int32_t n = hp.enc_n_blocks;
    if (n <= 1) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "titanet: n_blocks %d", n);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (const transcribe_status st = kv_i32_array(g, "titanet.block_repeats", n, hp.repeat); st != TRANSCRIBE_OK) {
        return st;
    }
    if (const transcribe_status st = kv_i32_array(g, "titanet.block_kernels", n, hp.kernel); st != TRANSCRIBE_OK) {
        return st;
    }
    hp.filters.assign(static_cast<size_t>(n), channels);
    hp.filters.back() = epilog;  // the epilog block widens
    hp.stride.assign(static_cast<size_t>(n), 1);
    hp.dilation.assign(static_cast<size_t>(n), 1);
    hp.residual.resize(static_cast<size_t>(n));
    for (int b = 0; b < n; ++b) {
        char name[64];
        std::snprintf(name, sizeof(name), "enc.b%d.res.conv.w", b);
        hp.residual[static_cast<size_t>(b)] = (gguf_find_tensor(g, name) >= 0) ? 1 : 0;
    }

    hp.enc_activation = "relu";
    hp.pooling        = "attention";

    // NeMo's AudioToMelSpectrogramPreprocessor as titanet-large configures it.
    // The published file carries the filterbank and window as tensors rather
    // than describing them, so the description comes from the architecture.
    hp.fe_num_mels    = hp.enc_feat_in;
    hp.fe_sample_rate = 16000;
    hp.fe_n_fft       = 512;
    hp.fe_win_length  = 400;
    hp.fe_hop_length  = 160;
    hp.fe_window      = "hann";
    hp.fe_normalize   = "per_feature";
    hp.fe_dither      = 1e-5f;
    if (const int64_t id = gguf_find_key(g, "titanet.n_fft"); id >= 0) {
        hp.fe_n_fft = static_cast<int32_t>(gguf_get_val_u32(g, id));
    }
    return TRANSCRIBE_OK;
}

transcribe_status read_titanet_hparams(const gguf_context * g, TitanetHParams & hp) {
    if (g == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // Both epsilons, before either spelling of everything else: the published
    // file states them and this one hard-coded them, and the encoder's 1e-3 is
    // the number that decides whether this model works at all.
    if (const int64_t id = gguf_find_key(g, "titanet.bn_eps_encoder"); id >= 0) {
        hp.bn_eps_block = gguf_get_val_f32(g, id);
    }
    if (const int64_t id = gguf_find_key(g, "titanet.bn_eps_decoder"); id >= 0) {
        hp.bn_eps_other = gguf_get_val_f32(g, id);
    }
    if (gguf_find_key(g, "stt.titanet.embedding_size") < 0 && gguf_find_key(g, "titanet.emb_dim") >= 0) {
        return read_published_hparams(g, hp);
    }

#define RD_U32(key, field)                                                          \
    if (const transcribe_status st = kv_u32(g, key, hp.field); st != TRANSCRIBE_OK) \
    return st
#define RD_F32(key, field)                                                          \
    if (const transcribe_status st = kv_f32(g, key, hp.field); st != TRANSCRIBE_OK) \
    return st
#define RD_STR(key, field)                                                          \
    if (const transcribe_status st = kv_str(g, key, hp.field); st != TRANSCRIBE_OK) \
    return st

    RD_U32("stt.titanet.embedding_size", embedding_size);
    RD_U32("stt.titanet.encoder.feat_in", enc_feat_in);
    RD_U32("stt.titanet.encoder.feat_out", enc_feat_out);
    RD_U32("stt.titanet.encoder.n_blocks", enc_n_blocks);
    RD_STR("stt.titanet.encoder.activation", enc_activation);
    RD_STR("stt.titanet.pooling", pooling);

    const int32_t n = hp.enc_n_blocks;
    if (n <= 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "titanet: n_blocks %d", n);
        return TRANSCRIBE_ERR_GGUF;
    }
#define RD_ARR(key, field)                                                                 \
    if (const transcribe_status st = kv_i32_array(g, key, n, hp.field); st != TRANSCRIBE_OK) \
    return st
    RD_ARR("stt.titanet.encoder.filters", filters);
    RD_ARR("stt.titanet.encoder.repeat", repeat);
    RD_ARR("stt.titanet.encoder.kernel", kernel);
    RD_ARR("stt.titanet.encoder.stride", stride);
    RD_ARR("stt.titanet.encoder.dilation", dilation);
    RD_ARR("stt.titanet.encoder.residual", residual);
#undef RD_ARR

    RD_U32("stt.frontend.num_mels", fe_num_mels);
    RD_U32("stt.frontend.sample_rate", fe_sample_rate);
    RD_U32("stt.frontend.n_fft", fe_n_fft);
    RD_U32("stt.frontend.win_length", fe_win_length);
    RD_U32("stt.frontend.hop_length", fe_hop_length);
    RD_STR("stt.frontend.window", fe_window);
    RD_STR("stt.frontend.normalize", fe_normalize);
    RD_F32("stt.frontend.dither", fe_dither);
    RD_F32("stt.frontend.pre_emphasis", fe_pre_emphasis);

#undef RD_U32
#undef RD_F32
#undef RD_STR

    if (hp.pooling != "attention") {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "titanet: pooling %s is not implemented", hp.pooling.c_str());
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.enc_activation != "relu") {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "titanet: activation %s is not implemented", hp.enc_activation.c_str());
        return TRANSCRIBE_ERR_GGUF;
    }
    // Every stride here is 1, and the pooling reduces the clip to one vector
    // whatever the length, so nothing downstream carries a frame rate. A
    // strided variant would change that and is worth failing on rather than
    // silently mis-locating an embedding in time.
    for (int i = 0; i < n; ++i) {
        if (hp.stride[static_cast<size_t>(i)] != 1 || hp.dilation[static_cast<size_t>(i)] != 1) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "titanet: block %d has stride %d dilation %d; only 1/1 is implemented",
                    i, hp.stride[static_cast<size_t>(i)], hp.dilation[static_cast<size_t>(i)]);
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    return TRANSCRIBE_OK;
}

namespace {

// Look up a tensor by name and validate its ne against the expected dims
// (fast-to-slow; pass -1 to skip a dim). Returns null and logs on failure.
// The same shape, ignoring axes of extent one. A pointwise convolution is a
// matrix and the two published files disagree about whether to store it as
// one: [1, in, out] here, [in, out] there, and [k, 1, C] against [k, C] for a
// depthwise kernel of width one. The elements are in the same order either
// way, so the graph reshapes what it is handed.
bool shape_ok(const ggml_tensor * t, int64_t ne0, int64_t ne1, int64_t ne2) {
    // "Do not check" is spelled -1 and only ever trails.
    if (ne1 < 0 && ne2 < 0) {
        return ne0 < 0 || t->ne[0] == ne0;
    }
    std::vector<int64_t> want;
    std::vector<int64_t> have;
    for (int64_t e : { ne0, ne1, ne2 }) {
        if (e > 1) {
            want.push_back(e);
        }
    }
    for (int i = 0; i < 3; ++i) {
        if (t->ne[i] > 1) {
            have.push_back(t->ne[i]);
        }
    }
    return want == have;
}

ggml_tensor * get_checked(ggml_context * ctx, const char * name, int64_t ne0, int64_t ne1, int64_t ne2) {
    ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (t == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "titanet: missing tensor %s", name);
        return nullptr;
    }
    if (!shape_ok(t, ne0, ne1, ne2)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "titanet: tensor %s shape mismatch: have [%lld,%lld,%lld] want [%lld,%lld,%lld]", name,
                (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) ne0, (long long) ne1,
                (long long) ne2);
        return nullptr;
    }
    return t;
}

// get_either finds a tensor under either published name.
ggml_tensor * get_either(ggml_context * ctx, const char * ours, const char * theirs, int64_t ne0, int64_t ne1,
                         int64_t ne2) {
    if (ggml_get_tensor(ctx, ours) == nullptr && ggml_get_tensor(ctx, theirs) != nullptr) {
        return get_checked(ctx, theirs, ne0, ne1, ne2);
    }
    return get_checked(ctx, ours, ne0, ne1, ne2);
}

}  // namespace

transcribe_status build_titanet_weights(ggml_context *         ctx,
                                        const TitanetHParams & hp,
                                        TitanetWeights &       w,
                                        const char *           tensor_prefix) {
    const char * pfx = (tensor_prefix != nullptr) ? tensor_prefix : "";
    char         name[160];

    char other[160];

// Every tensor is looked up under both published names. The prefix is for a
// bundle carrying this family's tensors beside another's.
#define GET(dst, nm, alt, e0, e1, e2)                                       \
    do {                                                                    \
        std::snprintf(name, sizeof(name), "%s%s", pfx, (nm));               \
        std::snprintf(other, sizeof(other), "%s%s", pfx, (alt));            \
        (dst) = get_either(ctx, name, other, (e0), (e1), (e2));             \
        if ((dst) == nullptr)                                               \
            return TRANSCRIBE_ERR_GGUF;                                     \
    } while (0)
#define GETF(dst, fmt, alt, e0, e1, e2, ...)                                \
    do {                                                                    \
        char slot[128];                                                     \
        char slot2[128];                                                    \
        std::snprintf(slot, sizeof(slot), fmt, __VA_ARGS__);                \
        std::snprintf(slot2, sizeof(slot2), alt, __VA_ARGS__);              \
        GET(dst, slot, slot2, e0, e1, e2);                                  \
    } while (0)

    w.blocks.resize(static_cast<size_t>(hp.enc_n_blocks));
    int64_t c_in = hp.enc_feat_in;
    for (int b = 0; b < hp.enc_n_blocks; ++b) {
        TitanetBlock & blk   = w.blocks[static_cast<size_t>(b)];
        const int64_t  c_out = hp.filters[static_cast<size_t>(b)];
        const int64_t  k     = hp.kernel[static_cast<size_t>(b)];
        const int      reps  = hp.repeat[static_cast<size_t>(b)];

        blk.reps.resize(static_cast<size_t>(reps));
        for (int r = 0; r < reps; ++r) {
            TitanetRepeat & rep = blk.reps[static_cast<size_t>(r)];
            // Only the first repeat changes the channel count; the rest run
            // at c_out.
            const int64_t in = (r == 0) ? c_in : c_out;
            GETF(rep.dw, "enc.blocks.%d.rep.%d.dw.weight", "enc.b%d.s%d.dw.w", k, 1, in, b, r);
            GETF(rep.pw, "enc.blocks.%d.rep.%d.pw.weight", "enc.b%d.s%d.pw.w", 1, in, c_out, b, r);
            GETF(rep.bn_w, "enc.blocks.%d.rep.%d.bn.weight", "enc.b%d.s%d.bn.w", c_out, -1, -1, b, r);
            GETF(rep.bn_b, "enc.blocks.%d.rep.%d.bn.bias", "enc.b%d.s%d.bn.b", c_out, -1, -1, b, r);
            GETF(rep.bn_rm, "enc.blocks.%d.rep.%d.bn.running_mean", "enc.b%d.s%d.bn.m", c_out, -1, -1, b, r);
            GETF(rep.bn_rv, "enc.blocks.%d.rep.%d.bn.running_var", "enc.b%d.s%d.bn.v", c_out, -1, -1, b, r);
        }

        GETF(blk.se_down, "enc.blocks.%d.se.down.weight", "enc.b%d.se.fc1.w", c_out, c_out / 8, -1, b);
        GETF(blk.se_up, "enc.blocks.%d.se.up.weight", "enc.b%d.se.fc2.w", c_out / 8, c_out, -1, b);

        if (hp.residual[static_cast<size_t>(b)] != 0) {
            GETF(blk.res_pw, "enc.blocks.%d.res.pw.weight", "enc.b%d.res.conv.w", 1, c_in, c_out, b);
            GETF(blk.res_bn_w, "enc.blocks.%d.res.bn.weight", "enc.b%d.res.bn.w", c_out, -1, -1, b);
            GETF(blk.res_bn_b, "enc.blocks.%d.res.bn.bias", "enc.b%d.res.bn.b", c_out, -1, -1, b);
            GETF(blk.res_bn_rm, "enc.blocks.%d.res.bn.running_mean", "enc.b%d.res.bn.m", c_out, -1, -1, b);
            GETF(blk.res_bn_rv, "enc.blocks.%d.res.bn.running_var", "enc.b%d.res.bn.v", c_out, -1, -1, b);
        }
        c_in = c_out;
    }

    const int64_t f = hp.enc_feat_out;
    GET(w.pool_attn0_w, "pool.attn.0.weight", "dec.asp.tdnn.w", 1, 3 * f, 128);
    GET(w.pool_attn0_b, "pool.attn.0.bias", "dec.asp.tdnn.b", 128, -1, -1);
    GET(w.pool_bn_w, "pool.attn.bn.weight", "dec.asp.bn.w", 128, -1, -1);
    GET(w.pool_bn_b, "pool.attn.bn.bias", "dec.asp.bn.b", 128, -1, -1);
    GET(w.pool_bn_rm, "pool.attn.bn.running_mean", "dec.asp.bn.m", 128, -1, -1);
    GET(w.pool_bn_rv, "pool.attn.bn.running_var", "dec.asp.bn.v", 128, -1, -1);
    GET(w.pool_attn1_w, "pool.attn.1.weight", "dec.asp.conv.w", 1, 128, f);
    GET(w.pool_attn1_b, "pool.attn.1.bias", "dec.asp.conv.b", f, -1, -1);

    GET(w.emb_bn_w, "emb.bn.weight", "dec.pool_bn.w", 2 * f, -1, -1);
    GET(w.emb_bn_b, "emb.bn.bias", "dec.pool_bn.b", 2 * f, -1, -1);
    GET(w.emb_bn_rm, "emb.bn.running_mean", "dec.pool_bn.m", 2 * f, -1, -1);
    GET(w.emb_bn_rv, "emb.bn.running_var", "dec.pool_bn.v", 2 * f, -1, -1);
    GET(w.emb_proj_w, "emb.proj.weight", "dec.fc.w", 1, 2 * f, hp.embedding_size);
    GET(w.emb_proj_b, "emb.proj.bias", "dec.fc.b", hp.embedding_size, -1, -1);

#undef GETF
#undef GET
    return TRANSCRIBE_OK;
}

}  // namespace transcribe::titanet
