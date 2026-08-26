// arch/fsmn_vad/weights.h - FSMN voice activity detector hparams and weights.
//
// Four FSMN layers between two pairs of affine transforms, over kaldi fbank
// features with LFR stacking. The whole model is 26 tensors.
//
// Tensor layout conventions (matched by scripts/convert-fsmn-vad.py):
//   - Linear weights: PyTorch [out, in] -> ggml ne [in, out].
//   - Memory: PyTorch [C, 1, lorder, 1] -> ggml ne [lorder, C], because a
//     depthwise convolution over time is what it is.
//   - CMVN: two vectors of d_input, applied by the host-side frontend.

#pragma once

#include "transcribe.h"

#include <cstdint>
#include <string>
#include <vector>

struct gguf_context;
struct ggml_context;
struct ggml_tensor;

namespace transcribe::fsmn_vad {

struct FsmnVadHParams {
    int32_t layers     = 0;  // 4
    int32_t input_dim  = 0;  // 400 = 80 mels x 5 stacked frames
    int32_t linear_dim = 0;  // 250
    int32_t proj_dim   = 0;  // 128
    int32_t lorder     = 0;  // 20 frames of memory, left only
    int32_t output_dim = 0;  // 248 HMM states
    // Which output means silence. A property of the checkpoint rather than of
    // the architecture, so it is read rather than assumed.
    int32_t silence_pdf = 0;

    // Frontend.
    int32_t     fe_sample_rate = 0;
    int32_t     fe_num_mels    = 0;
    int32_t     fe_win_length  = 0;
    int32_t     fe_hop_length  = 0;
    int32_t     fe_lfr_m       = 0;
    int32_t     fe_lfr_n       = 0;
    std::string fe_window;  // "hamming"

    // Milliseconds per frame, which the rows are measured in. LFR with n=1
    // keeps the frame rate of the fbank, so this is the hop.
    double frame_ms() const {
        return (fe_sample_rate > 0) ? 1000.0 * fe_hop_length * fe_lfr_n / fe_sample_rate : 0.0;
    }
};

// One FSMN layer: project down, remember the last lorder frames, project back.
struct FsmnLayer {
    ggml_tensor * linear   = nullptr;  // [linear_dim, proj_dim], no bias
    ggml_tensor * memory   = nullptr;  // [lorder, proj_dim] depthwise over time
    ggml_tensor * affine_w = nullptr;  // [proj_dim, linear_dim]
    ggml_tensor * affine_b = nullptr;  // [linear_dim]
};

struct FsmnVadWeights {
    ggml_tensor * cmvn_shift = nullptr;  // [input_dim]
    ggml_tensor * cmvn_scale = nullptr;

    ggml_tensor * in0_w = nullptr;  // [input_dim, 140]
    ggml_tensor * in0_b = nullptr;
    ggml_tensor * in1_w = nullptr;  // [140, linear_dim]
    ggml_tensor * in1_b = nullptr;

    std::vector<FsmnLayer> layers;

    ggml_tensor * out0_w = nullptr;  // [linear_dim, 140]
    ggml_tensor * out0_b = nullptr;
    ggml_tensor * out1_w = nullptr;  // [140, output_dim]
    ggml_tensor * out1_b = nullptr;
};

transcribe_status read_fsmn_vad_hparams(const gguf_context * gguf, FsmnVadHParams & hp);

// Look the tensors up by name and validate their shapes. tensor_prefix
// (nullable) is prepended to every name, so a bundle can carry these
// alongside another family's without a collision.
transcribe_status build_fsmn_vad_weights(ggml_context *         ctx_meta,
                                         const FsmnVadHParams & hp,
                                         FsmnVadWeights &       weights,
                                         const char *           tensor_prefix = nullptr);

}  // namespace transcribe::fsmn_vad
