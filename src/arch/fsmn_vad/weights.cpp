// arch/fsmn_vad/weights.cpp - hparam KV reader and weight catalog.

#include "weights.h"

#include "ggml.h"
#include "gguf.h"
#include "transcribe-log.h"

#include <cstdio>

namespace transcribe::fsmn_vad {

namespace {

transcribe_status kv_u32(const gguf_context * g, const char * key, int32_t & out) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "fsmn-vad: missing KV %s", key);
        return TRANSCRIBE_ERR_GGUF;
    }
    out = static_cast<int32_t>(gguf_get_val_u32(g, id));
    return TRANSCRIBE_OK;
}

transcribe_status kv_str(const gguf_context * g, const char * key, std::string & out) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "fsmn-vad: missing KV %s", key);
        return TRANSCRIBE_ERR_GGUF;
    }
    out = gguf_get_val_str(g, id);
    return TRANSCRIBE_OK;
}

ggml_tensor * get_checked(ggml_context * ctx, const char * name, int64_t ne0, int64_t ne1) {
    ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (t == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "fsmn-vad: missing tensor %s", name);
        return nullptr;
    }
    if ((ne0 >= 0 && t->ne[0] != ne0) || (ne1 >= 0 && t->ne[1] != ne1)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "fsmn-vad: tensor %s is [%lld,%lld], want [%lld,%lld]", name,
                (long long) t->ne[0], (long long) t->ne[1], (long long) ne0, (long long) ne1);
        return nullptr;
    }
    return t;
}

}  // namespace

transcribe_status read_fsmn_vad_hparams(const gguf_context * g, FsmnVadHParams & hp) {
    if (g == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
#define RD_U32(key, field)                                                          \
    if (const transcribe_status st = kv_u32(g, key, hp.field); st != TRANSCRIBE_OK) \
    return st

    RD_U32("stt.fsmn_vad.layers", layers);
    RD_U32("stt.fsmn_vad.input_dim", input_dim);
    RD_U32("stt.fsmn_vad.linear_dim", linear_dim);
    RD_U32("stt.fsmn_vad.proj_dim", proj_dim);
    RD_U32("stt.fsmn_vad.lorder", lorder);
    RD_U32("stt.fsmn_vad.output_dim", output_dim);
    RD_U32("stt.fsmn_vad.silence_pdf", silence_pdf);

    RD_U32("stt.frontend.sample_rate", fe_sample_rate);
    RD_U32("stt.frontend.num_mels", fe_num_mels);
    RD_U32("stt.frontend.win_length", fe_win_length);
    RD_U32("stt.frontend.hop_length", fe_hop_length);
    RD_U32("stt.frontend.lfr_m", fe_lfr_m);
    RD_U32("stt.frontend.lfr_n", fe_lfr_n);
#undef RD_U32
    if (const transcribe_status st = kv_str(g, "stt.frontend.window", hp.fe_window); st != TRANSCRIBE_OK) {
        return st;
    }

    if (hp.input_dim != hp.fe_num_mels * hp.fe_lfr_m) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "fsmn-vad: input_dim %d is not %d mels x %d stacked frames",
                hp.input_dim, hp.fe_num_mels, hp.fe_lfr_m);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.silence_pdf < 0 || hp.silence_pdf >= hp.output_dim) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "fsmn-vad: silence pdf %d is outside the %d outputs", hp.silence_pdf,
                hp.output_dim);
        return TRANSCRIBE_ERR_GGUF;
    }
    return TRANSCRIBE_OK;
}

transcribe_status build_fsmn_vad_weights(ggml_context *         ctx,
                                         const FsmnVadHParams & hp,
                                         FsmnVadWeights &       w,
                                         const char *           tensor_prefix) {
    const char * pfx = (tensor_prefix != nullptr) ? tensor_prefix : "";
    char         name[128];

#define GET(dst, nm, e0, e1)                                  \
    do {                                                      \
        std::snprintf(name, sizeof(name), "%s%s", pfx, (nm)); \
        (dst) = get_checked(ctx, name, (e0), (e1));           \
        if ((dst) == nullptr)                                 \
            return TRANSCRIBE_ERR_GGUF;                       \
    } while (0)

    // 140 is the width of both outer affine transforms and is not otherwise a
    // parameter: it is read off the tensor rather than declared.
    ggml_tensor * probe = ggml_get_tensor(ctx, "in.0.weight");
    if (probe == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "fsmn-vad: missing tensor in.0.weight");
        return TRANSCRIBE_ERR_GGUF;
    }
    const int64_t affine_dim = probe->ne[1];

    GET(w.cmvn_shift, "frontend.cmvn.shift", hp.input_dim, -1);
    GET(w.cmvn_scale, "frontend.cmvn.scale", hp.input_dim, -1);
    // Read back to host at load, so they have to be plain floats. A quantized
    // normalization vector is a smaller buffer than the read expects, and the
    // read runs off the end of it -- which is a crash rather than a bad
    // answer, but only because ggml checks.
    if (w.cmvn_shift->type != GGML_TYPE_F32 || w.cmvn_scale->type != GGML_TYPE_F32) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "fsmn-vad: normalization vectors must be F32, not %s",
                ggml_type_name(w.cmvn_shift->type));
        return TRANSCRIBE_ERR_GGUF;
    }

    GET(w.in0_w, "in.0.weight", hp.input_dim, affine_dim);
    GET(w.in0_b, "in.0.bias", affine_dim, -1);
    GET(w.in1_w, "in.1.weight", affine_dim, hp.linear_dim);
    GET(w.in1_b, "in.1.bias", hp.linear_dim, -1);

    w.layers.resize(static_cast<size_t>(hp.layers));
    for (int i = 0; i < hp.layers; ++i) {
        FsmnLayer & layer = w.layers[static_cast<size_t>(i)];
        char        slot[96];
#define GETL(dst, suffix, e0, e1)                                       \
    do {                                                                \
        std::snprintf(slot, sizeof(slot), "fsmn.%d.%s", i, suffix);     \
        GET(dst, slot, e0, e1);                                         \
    } while (0)
        GETL(layer.linear, "linear.weight", hp.linear_dim, hp.proj_dim);
        GETL(layer.memory, "memory.weight", hp.lorder, hp.proj_dim);
        GETL(layer.affine_w, "affine.weight", hp.proj_dim, hp.linear_dim);
        GETL(layer.affine_b, "affine.bias", hp.linear_dim, -1);
#undef GETL
    }

    GET(w.out0_w, "out.0.weight", hp.linear_dim, affine_dim);
    GET(w.out0_b, "out.0.bias", affine_dim, -1);
    GET(w.out1_w, "out.1.weight", affine_dim, hp.output_dim);
    GET(w.out1_b, "out.1.bias", hp.output_dim, -1);
#undef GET
    return TRANSCRIBE_OK;
}

}  // namespace transcribe::fsmn_vad
