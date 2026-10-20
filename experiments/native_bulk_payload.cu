#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/BFloat16.h>
#include <algorithm>

// Descriptor: nine pointers; old/prom counts and pitches; paid/rest/keep
// counts; paid output base; old cold length; exact integer timestamp.
__device__ uint16_t payload(const int64_t* row,int channel,int index,int head,int d,int dim){
  if(index<row[9])return reinterpret_cast<const uint16_t*>(row[channel])[(int64_t(head)*row[11]+index)*dim+d];
  index-=row[9];
  if(index<row[10])return reinterpret_cast<const uint16_t*>(row[2+channel])[(int64_t(head)*row[12]+index)*dim+d];
  const auto value=reinterpret_cast<const c10::BFloat16*>(row[4+channel])[int64_t(head)*row[13]*dim+d];
  // Match the ordinary new-token BF16 -> F32 -> BF16 path, including NaNs.
  const c10::BFloat16 converted(static_cast<float>(value));
  return converted.x;
}
__global__ void gather_phase(const int64_t* desc,int width,uint16_t* hot,uint32_t* meta,uint16_t* paid,
                            int layers,int heads,int keep_width,int paid_width,int rest_width,int dim,int paid_total){
  const int64_t off=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(off<int64_t(2)*layers*heads*keep_width*dim){
    const int d=off%dim,j=off/dim%keep_width,h=off/(int64_t(dim)*keep_width)%heads;
    const int l=off/(int64_t(dim)*keep_width*heads)%layers,ch=off/(int64_t(dim)*keep_width*heads*layers);
    const int64_t* row=desc+int64_t(l)*width;
    if(j<row[16])hot[off]=payload(row,ch,row[20+j],h,d,dim);
  }
  if(off<int64_t(2)*layers*heads*paid_width*dim){
    const int d=off%dim,j=off/dim%paid_width,h=off/(int64_t(dim)*paid_width)%heads;
    const int l=off/(int64_t(dim)*paid_width*heads)%layers,ch=off/(int64_t(dim)*paid_width*heads*layers);
    const int64_t* row=desc+int64_t(l)*width;
    if(j<row[14])paid[(int64_t(ch)*paid_total+row[17]+h*row[14]+j)*dim+d]=payload(row,ch,row[20+keep_width+j],h,d,dim);
  }
  if(off<int64_t(2)*layers*keep_width){
    const int j=off%keep_width,l=off/keep_width%layers,ch=off/(int64_t(keep_width)*layers);
    const int64_t* row=desc+int64_t(l)*width;
    if(j<row[16])meta[off]=reinterpret_cast<const uint32_t*>(row[6+ch])[row[20+j]];
  }
  const int touches=max(paid_width,rest_width);
  if(off<int64_t(layers)*touches){
    const int j=off%touches,l=off/touches;const int64_t* row=desc+int64_t(l)*width;
    auto stamp=reinterpret_cast<float*>(row[8]);const float time=static_cast<float>(row[19]);
    if(j<row[15])stamp[row[20+keep_width+paid_width+j]]=time;
    if(j<row[14])stamp[row[18]+j]=time;
  }
}
__global__ void scatter_phase(const int64_t* desc,int width,const uint16_t* hot,const uint32_t* meta,
                             int layers,int heads,int keep_width,int dim){
  const int64_t off=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(off<int64_t(2)*layers*heads*keep_width*dim){
    const int d=off%dim,j=off/dim%keep_width,h=off/(int64_t(dim)*keep_width)%heads;
    const int l=off/(int64_t(dim)*keep_width*heads)%layers,ch=off/(int64_t(dim)*keep_width*heads*layers);
    const int64_t* row=desc+int64_t(l)*width;
    if(j<row[16])reinterpret_cast<uint16_t*>(row[ch])[(int64_t(h)*row[11]+j)*dim+d]=hot[off];
  }
  if(off<int64_t(2)*layers*keep_width){
    const int j=off%keep_width,l=off/keep_width%layers,ch=off/(int64_t(keep_width)*layers);
    const int64_t* row=desc+int64_t(l)*width;
    if(j<row[16])reinterpret_cast<uint32_t*>(row[6+ch])[j]=meta[off];
  }
}
void run_cuda(torch::Tensor desc,torch::Tensor hot,torch::Tensor meta,torch::Tensor paid,
              int64_t first,int64_t keep_width,int64_t paid_width,int64_t rest_width){
  TORCH_CHECK(desc.is_cuda() && desc.scalar_type()==torch::kInt64 && desc.dim()==2 && desc.is_contiguous());
  TORCH_CHECK(hot.device()==desc.device() && hot.scalar_type()==torch::kBFloat16 && hot.is_contiguous() && hot.dim()==5 && hot.size(0)==2);
  const int layers=hot.size(1),heads=hot.size(2),dim=hot.size(4);
  TORCH_CHECK(layers>0 && layers<=8 && heads>0 && dim>0 && first>=0 && first+layers<=desc.size(0));
  TORCH_CHECK(keep_width>=0 && paid_width>=0 && rest_width>=0 && hot.size(3)==keep_width && desc.size(1)==20+keep_width+paid_width+rest_width);
  TORCH_CHECK(meta.device()==desc.device() && meta.scalar_type()==torch::kFloat32 && meta.is_contiguous() && meta.dim()==3 && meta.size(0)==2 && meta.size(1)==layers && meta.size(2)==keep_width);
  TORCH_CHECK(paid.device()==desc.device() && paid.scalar_type()==torch::kBFloat16 && paid.is_contiguous() && paid.dim()==4 && paid.size(0)==2 && paid.size(2)==1 && paid.size(3)==dim);
  c10::cuda::CUDAGuard guard(desc.device());auto stream=c10::cuda::getCurrentCUDAStream(desc.device().index()).stream();
  const int width=desc.size(1);const auto rows=desc.data_ptr<int64_t>()+first*width;
  const int64_t count=std::max({hot.numel(),int64_t(2)*layers*heads*paid_width*dim,meta.numel(),int64_t(layers)*std::max(paid_width,rest_width)});
  if(count){gather_phase<<<(count+255)/256,256,0,stream>>>(rows,width,reinterpret_cast<uint16_t*>(hot.data_ptr()),reinterpret_cast<uint32_t*>(meta.data_ptr()),reinterpret_cast<uint16_t*>(paid.data_ptr()),layers,heads,keep_width,paid_width,rest_width,dim,paid.size(1));C10_CUDA_KERNEL_LAUNCH_CHECK();}
  if(hot.numel()){scatter_phase<<<(hot.numel()+255)/256,256,0,stream>>>(rows,width,reinterpret_cast<const uint16_t*>(hot.data_ptr()),reinterpret_cast<const uint32_t*>(meta.data_ptr()),layers,heads,keep_width,dim);C10_CUDA_KERNEL_LAUNCH_CHECK();}
}
