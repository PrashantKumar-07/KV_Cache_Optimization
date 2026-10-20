#include <torch/extension.h>
void run_cuda(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run_cuda);}
