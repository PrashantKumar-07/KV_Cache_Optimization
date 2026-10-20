// Exact one-channel version of the existing paired pinned-host byte gather.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cassert>
#include <algorithm>
#include <cstdint>

__global__ void channel_kernel(const char* source, char* output,
    const int32_t* blocks, int64_t count, int64_t source_blocks,
    size_t row_bytes) {
  for (int64_t row=blockIdx.x; row<count; row+=gridDim.x) {
    const int32_t block=blocks[row];
    assert(block>=0 && block<source_blocks);
    const auto src=source+size_t(block)*row_bytes;
    auto dst=output+size_t(row)*row_bytes;
    for (size_t i=threadIdx.x; i<row_bytes/16; i+=blockDim.x)
      reinterpret_cast<uint4*>(dst)[i]=reinterpret_cast<const uint4*>(src)[i];
  }
}

void gather_channel_cuda(torch::Tensor pool, torch::Tensor blocks,
                         torch::Tensor output, int64_t channel) {
  TORCH_CHECK(pool.device().is_cpu() && pool.is_pinned() && pool.is_contiguous(),
              "Source must be contiguous pinned CPU pool");
  TORCH_CHECK(pool.dim()==4 && pool.size(0)==2 && pool.size(2)==1 && pool.size(3)>0,
              "Source must be [2, blocks, 1, dim]");
  TORCH_CHECK(channel==0 || channel==1, "Channel must be key0 or value1");
  TORCH_CHECK(output.is_cuda() && output.is_contiguous() && output.dim()==3,
              "Destination must be contiguous CUDA [heads, rows, dim]");
  TORCH_CHECK(blocks.is_cuda() && blocks.device()==output.device() &&
              blocks.scalar_type()==torch::kInt32 && blocks.is_contiguous() && blocks.dim()==2,
              "Source blocks must be contiguous CUDA int32 [heads, rows]");
  TORCH_CHECK(blocks.size(0)==output.size(0) && blocks.size(1)==output.size(1) &&
              output.size(2)==pool.size(3) && output.size(0)>0,
              "Block/destination geometry mismatch");
  TORCH_CHECK(pool.scalar_type()==output.scalar_type() &&
              (pool.scalar_type()==torch::kBFloat16 || pool.scalar_type()==torch::kFloat16 ||
               pool.scalar_type()==torch::kFloat32), "Channel dtype mismatch/unsupported");
  TORCH_CHECK(blocks.storage().unsafeGetStorageImpl()!=output.storage().unsafeGetStorageImpl(),
              "Source block indices alias destination");
  const size_t row_bytes=size_t(pool.size(3))*pool.element_size();
  TORCH_CHECK(row_bytes%16==0 && uintptr_t(pool.data_ptr())%16==0 &&
              uintptr_t(output.data_ptr())%16==0, "Channel rows require16byte alignment");
  const int64_t count=blocks.numel();
  if (count==0) return;
  c10::cuda::CUDAGuard guard(output.device());
  void* mapped=nullptr;
  C10_CUDA_CHECK(cudaHostGetDevicePointer(&mapped, pool.data_ptr(), 0));
  auto source=static_cast<const char*>(mapped)+size_t(channel)*pool.stride(0)*pool.element_size();
  channel_kernel<<<std::min<int64_t>(32,count),128,0,at::cuda::getCurrentCUDAStream()>>>(
      source,static_cast<char*>(output.data_ptr()),blocks.data_ptr<int32_t>(),count,
      pool.size(1),row_bytes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
