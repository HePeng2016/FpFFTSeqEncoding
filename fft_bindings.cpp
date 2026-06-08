#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "MixedRadixFFT.h"
#include <vector>
#include <cuComplex.h>

using cd = cuDoubleComplex;

// Your FFT class here (unchanged)
extern class MixedRadixFFT_GPU;

// Global singleton (simple approach)
static MixedRadixFFT_GPU* g_fft = nullptr;

// Initialize once
void init_fft(int N, std::vector<int> factors, int max_batch)
{
    if (g_fft == nullptr) {
        g_fft = new MixedRadixFFT_GPU(factors, max_batch);
    }
}

// Forward FFT wrapper
torch::Tensor fft_forward_cuda(torch::Tensor input)
{
    int64_t batch = input.size(0);
    int64_t N     = input.size(1);

    auto output = torch::empty_like(input);

    cd* d_input  = reinterpret_cast<cd*>(input.data_ptr<std::complex<double>>());
    cd* d_output = reinterpret_cast<cd*>(output.data_ptr<std::complex<double>>());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    g_fft->transform_batch_device_impl(
        d_input,
        d_output,
        batch,
        0,   // forward
        stream
    );

    return output;
}

// Inverse FFT wrapper
torch::Tensor fft_inverse_cuda(torch::Tensor input)
{
    int64_t batch = input.size(0);
    int64_t N     = input.size(1);

    auto output = torch::empty_like(input);

    cd* d_input  = reinterpret_cast<cd*>(input.data_ptr<std::complex<double>>());
    cd* d_output = reinterpret_cast<cd*>(output.data_ptr<std::complex<double>>());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    g_fft->transform_batch_device_impl(
        d_input,
        d_output,
        batch,
        1,   // inverse
        stream
    );

    return output;
}

// FFT backward (grad = N * IFFT)
torch::Tensor fft_backward_cuda(torch::Tensor grad_output)
{
    auto grad_input = fft_inverse_cuda(grad_output);

    int64_t N = grad_input.size(1);

    grad_input *= N;   // scaling

    return grad_input;
}

