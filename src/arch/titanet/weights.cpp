// arch/titanet/weights.cpp - TitaNet hparam KV reader and weight catalog.

#include "weights.h"

#include "ggml.h"
#include "gguf.h"
#include "transcribe-log.h"

#include <cstdio>
#include <string>

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

transcribe_status read_titanet_hparams(const gguf_context * g, TitanetHParams & hp) {
    if (g == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
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
ggml_tensor * get_checked(ggml_context * ctx, const char * name, int64_t ne0, int64_t ne1, int64_t ne2) {
    ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (t == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "titanet: missing tensor %s", name);
        return nullptr;
    }
    if ((ne0 >= 0 && t->ne[0] != ne0) || (ne1 >= 0 && t->ne[1] != ne1) || (ne2 >= 0 && t->ne[2] != ne2)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "titanet: tensor %s shape mismatch: have [%lld,%lld,%lld] want [%lld,%lld,%lld]", name,
                (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) ne0, (long long) ne1,
                (long long) ne2);
        return nullptr;
    }
    return t;
}

}  // namespace

transcribe_status build_titanet_weights(ggml_context *         ctx,
                                        const TitanetHParams & hp,
                                        TitanetWeights &       w,
                                        const char *           tensor_prefix) {
    const char * pfx = (tensor_prefix != nullptr) ? tensor_prefix : "";
    char         name[160];

#define GET(dst, nm, e0, e1, e2)                                 \
    do {                                                         \
        std::snprintf(name, sizeof(name), "%s%s", pfx, (nm));    \
        (dst) = get_checked(ctx, name, (e0), (e1), (e2));        \
        if ((dst) == nullptr)                                    \
            return TRANSCRIBE_ERR_GGUF;                          \
    } while (0)
#define GETF(dst, fmt, e0, e1, e2, ...)                             \
    do {                                                            \
        char slot[128];                                             \
        std::snprintf(slot, sizeof(slot), fmt, __VA_ARGS__);        \
        GET(dst, slot, e0, e1, e2);                                 \
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
            GETF(rep.dw, "enc.blocks.%d.rep.%d.dw.weight", k, 1, in, b, r);
            GETF(rep.pw, "enc.blocks.%d.rep.%d.pw.weight", 1, in, c_out, b, r);
            GETF(rep.bn_w, "enc.blocks.%d.rep.%d.bn.weight", c_out, -1, -1, b, r);
            GETF(rep.bn_b, "enc.blocks.%d.rep.%d.bn.bias", c_out, -1, -1, b, r);
            GETF(rep.bn_rm, "enc.blocks.%d.rep.%d.bn.running_mean", c_out, -1, -1, b, r);
            GETF(rep.bn_rv, "enc.blocks.%d.rep.%d.bn.running_var", c_out, -1, -1, b, r);
        }

        GETF(blk.se_down, "enc.blocks.%d.se.down.weight", c_out, c_out / 8, -1, b);
        GETF(blk.se_up, "enc.blocks.%d.se.up.weight", c_out / 8, c_out, -1, b);

        if (hp.residual[static_cast<size_t>(b)] != 0) {
            GETF(blk.res_pw, "enc.blocks.%d.res.pw.weight", 1, c_in, c_out, b);
            GETF(blk.res_bn_w, "enc.blocks.%d.res.bn.weight", c_out, -1, -1, b);
            GETF(blk.res_bn_b, "enc.blocks.%d.res.bn.bias", c_out, -1, -1, b);
            GETF(blk.res_bn_rm, "enc.blocks.%d.res.bn.running_mean", c_out, -1, -1, b);
            GETF(blk.res_bn_rv, "enc.blocks.%d.res.bn.running_var", c_out, -1, -1, b);
        }
        c_in = c_out;
    }

    const int64_t f = hp.enc_feat_out;
    GET(w.pool_attn0_w, "pool.attn.0.weight", 1, 3 * f, 128);
    GET(w.pool_attn0_b, "pool.attn.0.bias", 128, -1, -1);
    GET(w.pool_bn_w, "pool.attn.bn.weight", 128, -1, -1);
    GET(w.pool_bn_b, "pool.attn.bn.bias", 128, -1, -1);
    GET(w.pool_bn_rm, "pool.attn.bn.running_mean", 128, -1, -1);
    GET(w.pool_bn_rv, "pool.attn.bn.running_var", 128, -1, -1);
    GET(w.pool_attn1_w, "pool.attn.1.weight", 1, 128, f);
    GET(w.pool_attn1_b, "pool.attn.1.bias", f, -1, -1);

    GET(w.emb_bn_w, "emb.bn.weight", 2 * f, -1, -1);
    GET(w.emb_bn_b, "emb.bn.bias", 2 * f, -1, -1);
    GET(w.emb_bn_rm, "emb.bn.running_mean", 2 * f, -1, -1);
    GET(w.emb_bn_rv, "emb.bn.running_var", 2 * f, -1, -1);
    GET(w.emb_proj_w, "emb.proj.weight", 1, 2 * f, hp.embedding_size);
    GET(w.emb_proj_b, "emb.proj.bias", hp.embedding_size, -1, -1);

#undef GETF
#undef GET
    return TRANSCRIBE_OK;
}

}  // namespace transcribe::titanet
