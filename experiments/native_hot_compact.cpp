#include <torch/extension.h>

void compact_cuda(torch::Tensor pointers, torch::Tensor indices,
                  torch::Tensor sizes, torch::Tensor kv, torch::Tensor metadata);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("compact", &compact_cuda);
}
