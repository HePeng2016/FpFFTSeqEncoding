#include <torch/extension.h>

// forward declarations
torch::Tensor fft_forward_cuda(torch::Tensor input);
torch::Tensor fft_inverse_cuda(torch::Tensor input);
torch::Tensor fft_backward_cuda(torch::Tensor grad_output);

void init_fft(int N, std::vector<int> factors, int max_batch);

// -----------------------------
// FFT autograd
// -----------------------------
class FFTFunction : public torch::autograd::Function<FFTFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor input
    ) {
        return fft_forward_cuda(input);
    }

    static torch::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::tensor_list grad_outputs
    ) {
        auto grad_y = grad_outputs[0];

        // grad_x = N * IFFT(grad_y)
        auto grad_x = fft_backward_cuda(grad_y);

        return {grad_x};
    }
};

// -----------------------------
// IFFT autograd
// -----------------------------
class IFFTFunction : public torch::autograd::Function<IFFTFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor input
    ) {
        return fft_inverse_cuda(input);
    }

    static torch::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::tensor_list grad_outputs
    ) {
        auto grad_y = grad_outputs[0];

        // grad_x = FFT(grad_y)
        auto grad_x = fft_forward_cuda(grad_y);

        return {grad_x};
    }
};

// Python bindings

torch::Tensor fft_forward(torch::Tensor x) {
    return FFTFunction::apply(x);
}

torch::Tensor fft_inverse(torch::Tensor x) {
    return IFFTFunction::apply(x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init_fft", &init_fft);
    m.def("fft", &fft_forward);
    m.def("ifft", &fft_inverse);
}

