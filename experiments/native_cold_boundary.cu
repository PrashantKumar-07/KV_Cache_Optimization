#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
__global__ void rank_rows(const uint32_t* cold,const int32_t* indices,const int32_t* layers,uint32_t* out,int rows,int width,int capacity){
 int64_t off=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
 if(off<int64_t(rows)*width){int row=off/width;out[off]=cold[int64_t(layers[row])*capacity+indices[off]];}
}
void rank_cuda(torch::Tensor cold,torch::Tensor indices,torch::Tensor layers,torch::Tensor out){
 TORCH_CHECK(cold.is_cuda() && cold.scalar_type()==torch::kFloat32 && cold.dim()==2 && cold.is_contiguous());
 TORCH_CHECK(indices.device()==cold.device() && indices.scalar_type()==torch::kInt32 && indices.dim()==2 && indices.is_contiguous());
 TORCH_CHECK(layers.device()==cold.device() && layers.scalar_type()==torch::kInt32 && layers.dim()==1 && layers.is_contiguous() && layers.numel()==indices.size(0));
 TORCH_CHECK(out.device()==cold.device() && out.scalar_type()==torch::kFloat32 && out.sizes()==indices.sizes() && out.is_contiguous() && out.numel()>0);
 c10::cuda::CUDAGuard guard(cold.device());auto stream=c10::cuda::getCurrentCUDAStream(cold.device().index()).stream();
 rank_rows<<<(out.numel()+255)/256,256,0,stream>>>(reinterpret_cast<const uint32_t*>(cold.data_ptr()),indices.data_ptr<int32_t>(),layers.data_ptr<int32_t>(),reinterpret_cast<uint32_t*>(out.data_ptr()),indices.size(0),indices.size(1),cold.size(1));C10_CUDA_KERNEL_LAUNCH_CHECK();
}
__global__ void scatter_rows(const int64_t* desc,const uint16_t* source,int layers,int heads,int rows,int dim){
 int64_t off=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
 if(off<int64_t(2)*layers*heads*rows*dim){
  int d=off%dim,j=off/dim%rows,h=off/(int64_t(dim)*rows)%heads,l=off/(int64_t(dim)*rows*heads)%layers,ch=off/(int64_t(dim)*rows*heads*layers);
  auto row=desc+int64_t(l)*4;reinterpret_cast<uint16_t*>(row[ch])[(int64_t(h)*row[2]+row[3]+j)*dim+d]=source[off];
 }
}
void scatter_cuda(torch::Tensor desc,torch::Tensor source,int64_t layers,int64_t heads,int64_t rows){
 TORCH_CHECK(desc.is_cuda() && desc.scalar_type()==torch::kInt64 && desc.is_contiguous() && desc.dim()==2 && desc.size(0)==layers && desc.size(1)==4);
 TORCH_CHECK(source.device()==desc.device() && source.scalar_type()==torch::kBFloat16 && source.dim()==4 && source.is_contiguous() && source.size(0)==2 && source.size(1)==layers*heads*rows && source.size(2)==1);
 TORCH_CHECK(layers>0 && heads>0 && rows>0 && source.size(3)>0);
 c10::cuda::CUDAGuard guard(desc.device());auto stream=c10::cuda::getCurrentCUDAStream(desc.device().index()).stream();
 scatter_rows<<<(source.numel()+255)/256,256,0,stream>>>(desc.data_ptr<int64_t>(),reinterpret_cast<const uint16_t*>(source.data_ptr()),layers,heads,rows,source.size(3));C10_CUDA_KERNEL_LAUNCH_CHECK();
}
