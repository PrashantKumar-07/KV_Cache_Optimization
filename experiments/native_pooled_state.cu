#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

__global__ void prepare_kernel(float* arena,const int32_t* counts,int layers,int capacity) {
  const int64_t offset=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if (offset>=layers*capacity)return;
  const int layer=offset/capacity,row=offset%capacity;
  const int old=counts[layer*3],prom=counts[layer*3+1],added=counts[layer*3+2];
  if (row<old+prom) {
    if (row>=old) arena[offset]=0.0f;
    arena[layers*capacity+offset]=(row<old ? arena[layers*capacity+offset] : 0.0f)+1.0f;
  } else if (row<old+prom+added) {
    arena[offset]=0.0f;arena[layers*capacity+offset]=0.0f;
  }
}

__global__ void gather_kernel(const int64_t* sources,const int32_t* indices,const int32_t* pitches,
                             uint16_t* scratch,uint32_t* meta,int layers,int heads,int rows,int dim) {
  const int64_t offset=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if (offset<int64_t(2)*layers*heads*rows*dim) {
    const int d=offset%dim,row=offset/dim%rows;
    const int head=offset/(int64_t(dim)*rows)%heads;
    const int layer=offset/(int64_t(dim)*rows*heads)%layers;
    const int channel=offset/(int64_t(dim)*rows*heads*layers);
    const auto src=reinterpret_cast<const uint16_t*>(sources[layer*4+channel]);
    scratch[offset]=src[(int64_t(head)*pitches[layer]+indices[layer*rows+row])*dim+d];
  }
  if (offset<int64_t(2)*layers*rows) {
    const int row=offset%rows,layer=offset/rows%layers,channel=offset/(int64_t(rows)*layers);
    const auto src=reinterpret_cast<const uint32_t*>(sources[layer*4+2+channel]);
    meta[offset]=src[indices[layer*rows+row]];
  }
}

__global__ void scatter_kernel(const uint16_t* scratch,const uint32_t* meta,const int64_t* destinations,
                              uint32_t* arena,int layers,int heads,int rows,int dim,int total_layers,
                              int capacity,int first) {
  const int64_t offset=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if (offset<int64_t(2)*layers*heads*rows*dim) {
    const int d=offset%dim,row=offset/dim%rows;
    const int head=offset/(int64_t(dim)*rows)%heads;
    const int layer=offset/(int64_t(dim)*rows*heads)%layers;
    const int channel=offset/(int64_t(dim)*rows*heads*layers);
    auto dst=reinterpret_cast<uint16_t*>(destinations[layer*3+channel]);
    dst[(int64_t(head)*destinations[layer*3+2]+row)*dim+d]=scratch[offset];
  }
  if (offset<int64_t(2)*layers*rows) {
    const int row=offset%rows,layer=offset/rows%layers,channel=offset/(int64_t(rows)*layers);
    arena[(int64_t(channel)*total_layers+first+layer)*capacity+row]=meta[offset];
  }
}

void prepare_cuda(torch::Tensor arena,torch::Tensor counts) {
  TORCH_CHECK(arena.is_cuda() && arena.scalar_type()==torch::kFloat32 && arena.is_contiguous() &&
      arena.dim()==3 && arena.size(0)==2 && arena.size(1)>0 && arena.size(2)>0);
  TORCH_CHECK(counts.device()==arena.device() && counts.scalar_type()==torch::kInt32 &&
      counts.is_contiguous() && counts.dim()==2 && counts.size(0)==arena.size(1) && counts.size(1)==3);
  c10::cuda::CUDAGuard guard(arena.device());
  const int layers=arena.size(1),capacity=arena.size(2);
  prepare_kernel<<<(int64_t(layers)*capacity+255)/256,256,0,c10::cuda::getCurrentCUDAStream(arena.device().index()).stream()>>>(
      arena.data_ptr<float>(),counts.data_ptr<int32_t>(),layers,capacity);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void compact_cuda(torch::Tensor sources,torch::Tensor indices,torch::Tensor pitches,
                  torch::Tensor scratch,torch::Tensor scratch_meta,torch::Tensor destinations,
                  torch::Tensor metadata,int64_t first) {
  TORCH_CHECK(scratch.is_cuda() && scratch.scalar_type()==torch::kBFloat16 && scratch.is_contiguous() &&
      scratch.dim()==5 && scratch.size(0)==2 && scratch.size(1)>0 && scratch.size(2)>0 && scratch.size(4)>0);
  const auto device=scratch.device();const int layers=scratch.size(1),heads=scratch.size(2),rows=scratch.size(3),dim=scratch.size(4);
  TORCH_CHECK(layers<=8);
  for (auto t:{sources,indices,pitches,scratch_meta,destinations,metadata}) TORCH_CHECK(t.device()==device && t.is_contiguous());
  TORCH_CHECK(sources.scalar_type()==torch::kInt64 && sources.dim()==2 && sources.size(0)==layers && sources.size(1)==4);
  TORCH_CHECK(indices.scalar_type()==torch::kInt32 && indices.dim()==2 && indices.size(0)==layers && indices.size(1)==rows);
  TORCH_CHECK(pitches.scalar_type()==torch::kInt32 && pitches.dim()==1 && pitches.size(0)==layers);
  TORCH_CHECK(scratch_meta.scalar_type()==torch::kFloat32 && scratch_meta.dim()==3 && scratch_meta.size(0)==2 && scratch_meta.size(1)==layers && scratch_meta.size(2)==rows);
  TORCH_CHECK(destinations.scalar_type()==torch::kInt64 && destinations.dim()==2 && destinations.size(0)==layers && destinations.size(1)==3);
  TORCH_CHECK(metadata.scalar_type()==torch::kFloat32 && metadata.dim()==3 && metadata.size(0)==2 && metadata.size(2)>=rows && first>=0 && first+layers<=metadata.size(1));
  if (!rows)return;
  c10::cuda::CUDAGuard guard(device);auto stream=c10::cuda::getCurrentCUDAStream(device.index()).stream();
  const int64_t count=int64_t(2)*layers*heads*rows*dim;
  gather_kernel<<<(count+255)/256,256,0,stream>>>(sources.data_ptr<int64_t>(),indices.data_ptr<int32_t>(),pitches.data_ptr<int32_t>(),
      reinterpret_cast<uint16_t*>(scratch.data_ptr()),reinterpret_cast<uint32_t*>(scratch_meta.data_ptr()),layers,heads,rows,dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  scatter_kernel<<<(count+255)/256,256,0,stream>>>(reinterpret_cast<const uint16_t*>(scratch.data_ptr()),reinterpret_cast<const uint32_t*>(scratch_meta.data_ptr()),
      destinations.data_ptr<int64_t>(),reinterpret_cast<uint32_t*>(metadata.data_ptr()),layers,heads,rows,dim,metadata.size(1),metadata.size(2),first);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
