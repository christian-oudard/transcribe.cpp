// arch/fsmn_vad/weights.cpp - hparam KV reader and weight catalog.

#include "weights.h"

#include "ggml.h"
#include "gguf.h"
#include "transcribe-log.h"

#include <cstdio>

namespace transcribe::fsmn_vad {

namespace {

// Two spellings of this checkpoint are in circulation and both are read here.
//
// The first is what scripts/convert-fsmn-vad.py writes. The second is
// FunAudioLLM/fsmn-vad-GGUF, published before this port existed, which keeps
// FunASR's own checkpoint names and prefixes its hparams `vad.` rather than
// `stt.fsmn_vad.`. The weights in the two files are bit-identical, tensor by
// tensor, so what differs is spelling and the axis order of the memory
// kernel. Reading both means a user can fetch the published file rather than
// wait for anybody to publish another one.
//
// Ours is tried first, so a file that says something is believed over a
// default taken from the family.

// kv_u32 reads a hparam under either spelling.
transcribe_status kv_u32(const gguf_context * g, const char * key, const char * alt, int32_t & out) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0 && alt != nullptr) {
        id = gguf_find_key(g, alt);
    }
    if (id < 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "fsmn-vad: missing KV %s", key);
        return TRANSCRIBE_ERR_GGUF;
    }
    out = static_cast<int32_t>(gguf_get_val_u32(g, id));
    return TRANSCRIBE_OK;
}

// kv_u32_or is for what only one of the two spellings carries: the frontend's
// sampling and window, and which output means silence. There is one published
// checkpoint of this model and these are its values, so a file that omits them
// is not a file about a different model.
void kv_u32_or(const gguf_context * g, const char * key, int32_t fallback, int32_t & out) {
    const int64_t id = gguf_find_key(g, key);
    out              = (id < 0) ? fallback : static_cast<int32_t>(gguf_get_val_u32(g, id));
}

void kv_str_or(const gguf_context * g, const char * key, const char * fallback, std::string & out) {
    const int64_t id = gguf_find_key(g, key);
    out              = (id < 0) ? fallback : gguf_get_val_str(g, id);
}

// get_either finds a tensor under either spelling and checks its shape. The
// memory kernel is [lorder, C] in one file and [C, lorder] in the other, which
// is why a shape may be accepted either way round; the graph transposes what
// it is given.
ggml_tensor * get_either(ggml_context * ctx,
                         const char *   name,
                         const char *   alt,
                         int64_t        ne0,
                         int64_t        ne1,
                         bool           either_way = false) {
    ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (t == nullptr && alt != nullptr) {
        t = ggml_get_tensor(ctx, alt);
    }
    if (t == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "fsmn-vad: missing tensor %s", name);
        return nullptr;
    }
    const bool ok = (ne0 < 0 || t->ne[0] == ne0) && (ne1 < 0 || t->ne[1] == ne1);
    const bool swapped = either_way && ne0 >= 0 && ne1 >= 0 && t->ne[0] == ne1 && t->ne[1] == ne0;
    if (!ok && !swapped) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "fsmn-vad: tensor %s is [%lld,%lld], want [%lld,%lld]", t->name,
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
#define RD_U32(key, alt, field)                                                          \
    if (const transcribe_status st = kv_u32(g, key, alt, hp.field); st != TRANSCRIBE_OK) \
    return st

    RD_U32("stt.fsmn_vad.layers", "vad.fsmn_layers", layers);
    RD_U32("stt.fsmn_vad.input_dim", "vad.input_dim", input_dim);
    RD_U32("stt.fsmn_vad.linear_dim", "vad.linear_dim", linear_dim);
    RD_U32("stt.fsmn_vad.proj_dim", "vad.proj_dim", proj_dim);
    RD_U32("stt.fsmn_vad.lorder", "vad.lorder", lorder);
    RD_U32("stt.fsmn_vad.output_dim", "vad.output_dim", output_dim);

    RD_U32("stt.frontend.num_mels", "vad.n_mels", fe_num_mels);
    RD_U32("stt.frontend.lfr_m", "vad.lfr_m", fe_lfr_m);
    RD_U32("stt.frontend.lfr_n", "vad.lfr_n", fe_lfr_n);
#undef RD_U32

    // What only one of the two spellings carries. The values are this
    // checkpoint's: 16 kHz, a 25 ms hamming window every 10 ms, and pdf 0 for
    // silence. A file that states them is believed over these.
    kv_u32_or(g, "stt.fsmn_vad.silence_pdf", 0, hp.silence_pdf);
    kv_u32_or(g, "stt.frontend.sample_rate", 16000, hp.fe_sample_rate);
    kv_u32_or(g, "stt.frontend.win_length", 400, hp.fe_win_length);
    kv_u32_or(g, "stt.frontend.hop_length", 160, hp.fe_hop_length);
    kv_str_or(g, "stt.frontend.window", "hamming", hp.fe_window);

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
    char         other[128];

#define GET(dst, nm, alt, e0, e1)                                 \
    do {                                                          \
        std::snprintf(name, sizeof(name), "%s%s", pfx, (nm));     \
        std::snprintf(other, sizeof(other), "%s%s", pfx, (alt));  \
        (dst) = get_either(ctx, name, other, (e0), (e1));         \
        if ((dst) == nullptr)                                     \
            return TRANSCRIBE_ERR_GGUF;                           \
    } while (0)

    // 140 is the width of both outer affine transforms and is not otherwise a
    // parameter: it is read off the tensor rather than declared.
    ggml_tensor * probe = get_either(ctx, "in.0.weight", "encoder.in_linear1.linear.weight", -1, -1);
    if (probe == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }
    const int64_t affine_dim = probe->ne[1];

    GET(w.cmvn_shift, "frontend.cmvn.shift", "cmvn.shift", hp.input_dim, -1);
    GET(w.cmvn_scale, "frontend.cmvn.scale", "cmvn.scale", hp.input_dim, -1);
    // Read back to host at load, so they have to be plain floats. A quantized
    // normalization vector is a smaller buffer than the read expects, and the
    // read runs off the end of it -- which is a crash rather than a bad
    // answer, but only because ggml checks.
    if (w.cmvn_shift->type != GGML_TYPE_F32 || w.cmvn_scale->type != GGML_TYPE_F32) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "fsmn-vad: normalization vectors must be F32, not %s",
                ggml_type_name(w.cmvn_shift->type));
        return TRANSCRIBE_ERR_GGUF;
    }

    GET(w.in0_w, "in.0.weight", "encoder.in_linear1.linear.weight", hp.input_dim, affine_dim);
    GET(w.in0_b, "in.0.bias", "encoder.in_linear1.linear.bias", affine_dim, -1);
    GET(w.in1_w, "in.1.weight", "encoder.in_linear2.linear.weight", affine_dim, hp.linear_dim);
    GET(w.in1_b, "in.1.bias", "encoder.in_linear2.linear.bias", hp.linear_dim, -1);

    w.layers.resize(static_cast<size_t>(hp.layers));
    for (int i = 0; i < hp.layers; ++i) {
        FsmnLayer & layer = w.layers[static_cast<size_t>(i)];
        char        slot[96];
        char theirs[96];
#define GETL(dst, suffix, alt, e0, e1)                                        \
    do {                                                                      \
        std::snprintf(slot, sizeof(slot), "fsmn.%d.%s", i, suffix);           \
        std::snprintf(theirs, sizeof(theirs), "encoder.fsmn.%d.%s", i, alt);  \
        GET(dst, slot, theirs, e0, e1);                                       \
    } while (0)
        GETL(layer.linear, "linear.weight", "linear.linear.weight", hp.linear_dim, hp.proj_dim);
        GETL(layer.affine_w, "affine.weight", "affine.linear.weight", hp.proj_dim, hp.linear_dim);
        GETL(layer.affine_b, "affine.bias", "affine.linear.bias", hp.linear_dim, -1);
#undef GETL
        // The memory kernel, whose two published spellings disagree about
        // which axis is time. Either is accepted and the graph transposes.
        std::snprintf(slot, sizeof(slot), "%sfsmn.%d.memory.weight", pfx, i);
        std::snprintf(theirs, sizeof(theirs), "%sencoder.fsmn.%d.fsmn_block.conv_left.weight", pfx, i);
        layer.memory = get_either(ctx, slot, theirs, hp.lorder, hp.proj_dim, /*either_way=*/true);
        if (layer.memory == nullptr) {
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    GET(w.out0_w, "out.0.weight", "encoder.out_linear1.linear.weight", hp.linear_dim, affine_dim);
    GET(w.out0_b, "out.0.bias", "encoder.out_linear1.linear.bias", affine_dim, -1);
    GET(w.out1_w, "out.1.weight", "encoder.out_linear2.linear.weight", affine_dim, hp.output_dim);
    GET(w.out1_b, "out.1.bias", "encoder.out_linear2.linear.bias", hp.output_dim, -1);
#undef GET
    return TRANSCRIBE_OK;
}

}  // namespace transcribe::fsmn_vad
