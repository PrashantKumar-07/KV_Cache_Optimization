#include <torch/extension.h>
void prepare_cuda(torch::Tensor arena,torch::Tensor counts);
void compact_cuda(torch::Tensor sources,torch::Tensor indices,torch::Tensor pitches,
                  torch::Tensor scratch,torch::Tensor scratch_meta,torch::Tensor destinations,
                  torch::Tensor metadata, int64_t first);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("prepare",&prepare_cuda);
  m.def("compact",&compact_cuda);
}
