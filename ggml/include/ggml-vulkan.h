#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_VK_NAME "Vulkan"
#define GGML_VK_MAX_DEVICES 16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_vk_init(size_t dev_num);

GGML_BACKEND_API bool ggml_backend_is_vk(ggml_backend_t backend);
GGML_BACKEND_API int  ggml_backend_vk_get_device_count(void);
GGML_BACKEND_API void ggml_backend_vk_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_vk_get_device_memory(int device, size_t * free, size_t * total);
// How many compute pipelines this device has compiled so far. Pipelines are
// compiled on first use, so this rises the first time a graph needs a shape
// no earlier graph did, and stops rising once a workload is warm. Callers use
// it to tell "this run was slow because the model is big" from "this run was
// slow because it was still compiling". 0 if the device was never initialized.
GGML_BACKEND_API size_t ggml_backend_vk_get_device_compiled_pipelines(int device);
// The name of the pipeline this device compiled at `index`, in compile order,
// or NULL past the end. Names carry the variant, so they say what a new shape
// actually needed: which matmul tile size it selected, and whether its
// dimensions let it take the aligned path. The string belongs to the device
// and stays valid until the device is freed.
GGML_BACKEND_API const char * ggml_backend_vk_get_device_compiled_pipeline_name(int device, size_t index);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_num);
// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_host_buffer_type(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_vk_reg(void);

#ifdef  __cplusplus
}
#endif
