#include <torch/extension.h>
void rank_cuda(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor);
void scatter_cuda(torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("rank",&rank_cuda);m.def("scatter",&scatter_cuda);}
