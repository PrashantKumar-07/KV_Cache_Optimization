#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

__global__ void compact_kernel(const int64_t* pointers, const int32_t* indices,
                               const int32_t* sizes, uint16_t* kv, float* metadata,
                               int layers, int heads, int rows, int dim) {
  const int64_t offset = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t kv_count = int64_t(2) * layers * heads * rows * dim;
  if (offset < kv_count) {
    const int d = offset % dim;
    const int row = offset / dim % rows;
    const int head = offset / (int64_t(dim) * rows) % heads;
    const int layer = offset / (int64_t(dim) * rows * heads) % layers;
    const int channel = offset / (int64_t(dim) * rows * heads * layers);
    const auto source = reinterpret_cast<const uint16_t*>(pointers[layer * 4 + channel]);
    const int old_row = indices[layer * rows + row];
    kv[offset] = source[(int64_t(head) * sizes[layer] + old_row) * dim + d];
  }
  const int64_t metadata_count = int64_t(2) * layers * rows;
  if (offset < metadata_count) {
    const int row = offset % rows;
    const int layer = offset / rows % layers;
    const int channel = offset / (int64_t(rows) * layers);
    const auto source = reinterpret_cast<const float*>(pointers[layer * 4 + 2 + channel]);
    metadata[offset] = source[indices[layer * rows + row]];
  }
}

void compact_cuda(torch::Tensor pointers, torch::Tensor indices,
                  torch::Tensor sizes, torch::Tensor kv, torch::Tensor metadata) {
  TORCH_CHECK(kv.is_cuda() && kv.scalar_type() == torch::kBFloat16 && kv.is_contiguous());
  TORCH_CHECK(kv.dim() == 5 && kv.size(0) == 2);
  const auto device = kv.device();
  TORCH_CHECK(pointers.device() == device && indices.device() == device &&
              sizes.device() == device && metadata.device() == device);
  const int layers = kv.size(1), heads = kv.size(2), rows = kv.size(3), dim = kv.size(4);
  TORCH_CHECK(pointers.scalar_type() == torch::kInt64 && pointers.is_contiguous() &&
              pointers.dim() == 2 && pointers.size(0) == layers && pointers.size(1) == 4);
  TORCH_CHECK(indices.scalar_type() == torch::kInt32 && indices.is_contiguous() &&
              indices.dim() == 2 && indices.size(0) == layers && indices.size(1) == rows);
  TORCH_CHECK(sizes.scalar_type() == torch::kInt32 && sizes.is_contiguous() && sizes.numel() == layers);
  TORCH_CHECK(metadata.scalar_type() == torch::kFloat32 && metadata.is_contiguous() &&
              metadata.dim() == 3 && metadata.size(0) == 2 && metadata.size(1) == layers &&
              metadata.size(2) == rows);
  if (kv.numel() == 0) return;
  c10::cuda::CUDAGuard guard(device);
  compact_kernel<<<(kv.numel() + 255) / 256, 256, 0,
      c10::cuda::getCurrentCUDAStream(device.index()).stream()>>>(
      pointers.data_ptr<int64_t>(), indices.data_ptr<int32_t>(), sizes.data_ptr<int32_t>(),
      reinterpret_cast<uint16_t*>(kv.data_ptr()), metadata.data_ptr<float>(), layers, heads, rows, dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
