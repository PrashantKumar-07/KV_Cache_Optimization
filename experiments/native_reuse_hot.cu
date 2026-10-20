#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

// The existing pybind declaration is reused. Its fourth argument describes
// heads/dimension with zero elements; destination pointers are in columns4/5.
__global__ void reuse_kernel(const int64_t* pointers, const int32_t* indices,
                            const int32_t* sizes, uint32_t* metadata,
                            int layers, int heads, int rows, int dim) {
  const int64_t offset=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  const int64_t kv_count=int64_t(2)*layers*heads*rows*dim;
  if (offset<kv_count) {
    const int d=offset%dim, row=offset/dim%rows;
    const int head=offset/(int64_t(dim)*rows)%heads;
    const int layer=offset/(int64_t(dim)*rows*heads)%layers;
    const int channel=offset/(int64_t(dim)*rows*heads*layers);
    const auto source=reinterpret_cast<const uint16_t*>(pointers[layer*6+channel]);
    const auto destination=reinterpret_cast<uint16_t*>(pointers[layer*6+4+channel]);
    destination[(int64_t(head)*sizes[layer*2+1]+row)*dim+d]=
        source[(int64_t(head)*sizes[layer*2]+indices[layer*rows+row])*dim+d];
  }
  if (offset<int64_t(2)*layers*rows) {
    const int row=offset%rows, layer=offset/rows%layers;
    const int channel=offset/(int64_t(rows)*layers);
    const auto source=reinterpret_cast<const uint32_t*>(pointers[layer*6+2+channel]);
    metadata[offset]=source[indices[layer*rows+row]];
  }
}

void compact_cuda(torch::Tensor pointers, torch::Tensor indices,
                  torch::Tensor sizes, torch::Tensor geometry, torch::Tensor metadata) {
  TORCH_CHECK(geometry.is_cuda() && geometry.scalar_type()==torch::kBFloat16 &&
      geometry.is_contiguous() && geometry.dim()==5 && geometry.size(0)==2 && geometry.size(3)==0);
  const auto device=geometry.device();
  const int layers=geometry.size(1), heads=geometry.size(2), dim=geometry.size(4);
  TORCH_CHECK(layers>0 && heads>0 && dim>0);
  TORCH_CHECK(pointers.device()==device && indices.device()==device &&
      sizes.device()==device && metadata.device()==device);
  TORCH_CHECK(pointers.scalar_type()==torch::kInt64 && pointers.is_contiguous() &&
      pointers.dim()==2 && pointers.size(0)==layers && pointers.size(1)==6);
  TORCH_CHECK(indices.scalar_type()==torch::kInt32 && indices.is_contiguous() &&
      indices.dim()==2 && indices.size(0)==layers);
  const int rows=indices.size(1);
  TORCH_CHECK(sizes.scalar_type()==torch::kInt32 && sizes.is_contiguous() &&
      sizes.dim()==2 && sizes.size(0)==layers && sizes.size(1)==2);
  TORCH_CHECK(metadata.scalar_type()==torch::kFloat32 && metadata.is_contiguous() &&
      metadata.dim()==3 && metadata.size(0)==2 && metadata.size(1)==layers && metadata.size(2)==rows);
  if (rows==0)return;
  c10::cuda::CUDAGuard guard(device);
  const int64_t count=int64_t(2)*layers*heads*rows*dim;
  reuse_kernel<<<(count+255)/256,256,0,c10::cuda::getCurrentCUDAStream(device.index()).stream()>>>(
      pointers.data_ptr<int64_t>(),indices.data_ptr<int32_t>(),sizes.data_ptr<int32_t>(),
      reinterpret_cast<uint32_t*>(metadata.data_ptr<float>()),layers,heads,rows,dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
