// arch/titanet/weights.h - TitaNet hparams and weight slots.
//
// TitaNet is an `encoder-embedding`: five ContextNet blocks of separable
// convolutions with squeeze-excite, attentive statistics pooling, and a
// projection to one 192-dim vector per clip. No tokenizer, no decoder, no
// text.
//
// Tensor layout conventions (matched by scripts/convert-titanet.py):
//   - Depthwise conv:  PyTorch [C, 1, K]      -> ggml ne [K, 1, C].
//   - Pointwise conv:  PyTorch [Cout, Cin, 1] -> ggml ne [1, Cin, Cout].
//   - Linear weights:  PyTorch [out, in]      -> ggml ne [in, out].
//   - BatchNorm arrives as four separate [C] tensors and is folded into a
//     scale and a shift at load; see fuse_batch_norms in model.cpp.

#pragma once

#include "transcribe.h"

#include <cstdint>
#include <string>
#include <vector>

struct gguf_context;
struct ggml_context;
struct ggml_tensor;

namespace transcribe::titanet {

struct TitanetHParams {
    int32_t embedding_size = 0;  // 192
    int32_t enc_feat_in    = 0;  // 80, the mel bins
    int32_t enc_feat_out   = 0;  // 3072, the last block's channels
    int32_t enc_n_blocks   = 0;  // 5

    // Per-block, all of length enc_n_blocks.
    std::vector<int32_t> filters;
    std::vector<int32_t> repeat;
    std::vector<int32_t> kernel;
    std::vector<int32_t> stride;
    std::vector<int32_t> dilation;
    std::vector<int32_t> residual;

    std::string enc_activation;  // "relu"
    std::string pooling;         // "attention"

    // Frontend (mel feature extractor).
    int32_t     fe_num_mels    = 0;
    int32_t     fe_sample_rate = 0;
    int32_t     fe_n_fft       = 0;
    int32_t     fe_win_length  = 0;
    int32_t     fe_hop_length  = 0;
    std::string fe_window;     // "hann"
    std::string fe_normalize;  // "per_feature"
    float       fe_dither       = 0.0f;
    float       fe_pre_emphasis = 0.0f;
};

// One [depthwise, pointwise, batchnorm] group inside a block. A block runs
// `repeat` of these; every one but the last is followed by ReLU.
struct TitanetRepeat {
    ggml_tensor * dw     = nullptr;  // [K, 1, C_in]
    ggml_tensor * pw     = nullptr;  // [1, C_in, C_out]
    ggml_tensor * bn_w   = nullptr;  // [C_out]
    ggml_tensor * bn_b   = nullptr;
    ggml_tensor * bn_rm  = nullptr;
    ggml_tensor * bn_rv  = nullptr;
    ggml_tensor * scale  = nullptr;  // folded BN, filled at load
    ggml_tensor * shift  = nullptr;
};

struct TitanetBlock {
    std::vector<TitanetRepeat> reps;

    // Squeeze-excite (jasper.SqueezeExcite, reduction 8, both linears
    // biasless): mean over the clip, down, ReLU, up, sigmoid, rescale.
    ggml_tensor * se_down = nullptr;  // [C, C/8]
    ggml_tensor * se_up   = nullptr;  // [C/8, C]

    // Skip path, on blocks that declare a residual: pointwise then BN.
    ggml_tensor * res_pw    = nullptr;  // [1, C_in, C_out]
    ggml_tensor * res_bn_w  = nullptr;
    ggml_tensor * res_bn_b  = nullptr;
    ggml_tensor * res_bn_rm = nullptr;
    ggml_tensor * res_bn_rv = nullptr;
    ggml_tensor * res_scale = nullptr;  // folded BN, filled at load
    ggml_tensor * res_shift  = nullptr;
};

struct TitanetWeights {
    std::vector<TitanetBlock> blocks;

    // Attentive statistics pooling. attn_0 takes [x, mean, std] (3 * C_in)
    // to 128, attn_1 takes 128 back to C_in, one softmax weight per channel.
    ggml_tensor * pool_attn0_w    = nullptr;  // [1, 3 * feat_out, 128]
    ggml_tensor * pool_attn0_b    = nullptr;  // [128]
    ggml_tensor * pool_bn_w       = nullptr;  // [128]
    ggml_tensor * pool_bn_b       = nullptr;
    ggml_tensor * pool_bn_rm      = nullptr;
    ggml_tensor * pool_bn_rv      = nullptr;
    ggml_tensor * pool_bn_scale   = nullptr;  // folded BN, filled at load
    ggml_tensor * pool_bn_shift   = nullptr;
    ggml_tensor * pool_attn1_w    = nullptr;  // [1, 128, feat_out]
    ggml_tensor * pool_attn1_b    = nullptr;  // [feat_out]

    // Embedding: BatchNorm over the 6144 pooled statistics, then Linear.
    ggml_tensor * emb_bn_w     = nullptr;  // [2 * feat_out]
    ggml_tensor * emb_bn_b     = nullptr;
    ggml_tensor * emb_bn_rm    = nullptr;
    ggml_tensor * emb_bn_rv    = nullptr;
    ggml_tensor * emb_bn_scale = nullptr;  // folded BN, filled at load
    ggml_tensor * emb_bn_shift = nullptr;
    ggml_tensor * emb_proj_w   = nullptr;  // [1, 2 * feat_out, embedding_size]
    ggml_tensor * emb_proj_b   = nullptr;  // [embedding_size]
};

// Read every required stt.titanet.* / stt.frontend.* KV into hp.
transcribe_status read_titanet_hparams(const gguf_context * gguf, TitanetHParams & hp);

// Look up the tensors by name in ctx_meta, validate their shapes against hp,
// and store borrowed pointers. Returns TRANSCRIBE_ERR_GGUF, naming the
// tensor, on anything missing or mis-shaped.
//
// tensor_prefix (nullable, standalone default "") is prepended to every name,
// so a bundle GGUF can carry these weights under a "titanet." namespace
// without colliding with the host family's own catalog.
transcribe_status build_titanet_weights(ggml_context *         ctx_meta,
                                        const TitanetHParams & hp,
                                        TitanetWeights &       weights,
                                        const char *           tensor_prefix = nullptr);

}  // namespace transcribe::titanet
