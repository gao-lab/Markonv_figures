#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> markonv_cuda_forward(
    torch::Tensor input,
    torch::Tensor Kernel_Full_4DTensor,
    torch::Tensor output,
    int stride);

std::vector<torch::Tensor> markonv_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor Kernel_Full_4DTensor,
    int stride);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> markonv_forward(
    torch::Tensor input,
    torch::Tensor Kernel_Full_4DTensor,
    torch::Tensor output,
    int stride
) {
  CHECK_INPUT(input);
  CHECK_INPUT(Kernel_Full_4DTensor);
  CHECK_INPUT(output);

  TORCH_CHECK(stride > 0, "stride " + std::to_string(stride) + " must be positive");

  
  return markonv_cuda_forward(input, Kernel_Full_4DTensor, output, stride);
}

std::vector<torch::Tensor> markonv_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor Kernel_Full_4DTensor,
    int stride) {
  CHECK_INPUT(grad_output);
  CHECK_INPUT(input);
  CHECK_INPUT(Kernel_Full_4DTensor);

  TORCH_CHECK(stride > 0, "stride " + std::to_string(stride) + " must be positive");

  return markonv_cuda_backward(
      grad_output,
      input,
      Kernel_Full_4DTensor,
      stride);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &markonv_forward, "Markonv forward (CUDA)");
  m.def("backward", &markonv_backward, "Markonv backward (CUDA)");
}



