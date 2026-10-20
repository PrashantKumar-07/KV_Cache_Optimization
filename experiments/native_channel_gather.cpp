#include <torch/extension.h>

void gather_channel_cuda(torch::Tensor pool, torch::Tensor blocks,
                         torch::Tensor output, int64_t channel);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("gather", &gather_channel_cuda);
}
