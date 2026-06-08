#include <cuda_runtime.h>
#include <cuComplex.h>

#include <vector>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <cstdlib>
#include <algorithm>

using cd = cuDoubleComplex;

constexpr double PI = 3.141592653589793238462643383279502884;

#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err__ = (call);                                         \
        if (err__ != cudaSuccess) {                                         \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__    \
                      << " : " << cudaGetErrorString(err__) << std::endl;   \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                  \
    } while (0)

__host__ __device__
inline cd cadd(cd a, cd b) {
    return make_cuDoubleComplex(
        cuCreal(a) + cuCreal(b),
        cuCimag(a) + cuCimag(b)
    );
}

__host__ __device__
inline cd cmul(cd a, cd b) {
    return make_cuDoubleComplex(
        cuCreal(a) * cuCreal(b) - cuCimag(a) * cuCimag(b),
        cuCreal(a) * cuCimag(b) + cuCimag(a) * cuCreal(b)
    );
}


__host__ __device__
inline cd cconj(cd a) {
    return make_cuDoubleComplex(
        cuCreal(a),
        -cuCimag(a)
    );
}

__device__
inline cd get_twiddle(
    const cd* W,
    int idx,
    int N,
    int inverse
) {
    idx %= N;
    if (idx < 0) idx += N;

    cd w = W[idx];

    if (inverse) {
        return cconj(w);   // exp(+2*pi*i*k/N)
    } else {
        return w;          // exp(-2*pi*i*k/N)
    }
}





__global__
void init_twiddles(cd* W, int N) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;

    double angle = -2.0 * PI * static_cast<double>(k) / static_cast<double>(N);
    W[k] = make_cuDoubleComplex(std::cos(angle), std::sin(angle));
}

__global__
void split_stage_shared_kernel(
    const cd* __restrict__ input,
    cd* __restrict__ output,
    const cd* __restrict__ W,
    int Ntotal,
    int sub_n,
    int radix,
    int prim1,
    int batch,
    int inverse  
) {
    extern __shared__ cd sdata[];

    int task = blockIdx.x;

    if (task >= batch * prim1) return;

    int b = task / prim1;
    int c = task % prim1;

    int tid = threadIdx.x;

    int input_base = b * sub_n;

    // Load x[c + m * prim1] into shared memory.
    for (int m = tid; m < radix; m += blockDim.x) {
        sdata[m] = input[input_base + c + m * prim1];
    }

    __syncthreads();

    // Each thread computes one or more r0 outputs.
    for (int r0 = tid; r0 < radix; r0 += blockDim.x) {
        cd sum = make_cuDoubleComplex(0.0, 0.0);

        for (int m = 0; m < radix; ++m) {
            // exp(-2*pi*i*r0*m/radix)
            int idx1 = (r0 * m * (Ntotal / radix)) % Ntotal;
            cd w1 = get_twiddle(W, idx1, Ntotal, inverse);
            sum = cadd(sum, cmul(sdata[m], w1));
        }

        // CPU twiddle:
        // U[r0, c] *= exp(-2*pi*i*r0*c/sub_n)
        int idx2 = (r0 * c * (Ntotal / sub_n)) % Ntotal;
        cd tw = get_twiddle(W, idx2, Ntotal, inverse);

        sum = cmul(sum, tw);

        // Output layout:
        // rows become independent subproblems for next stage.
        //
        // CPU U[row * prim1 + col]
        //
        // New batch index = b * radix + r0
        output[(b * radix + r0) * prim1 + c] = sum;
    }
}

__global__
void split_stage_global_kernel(
    const cd* __restrict__ input,
    cd* __restrict__ output,
    const cd* __restrict__ W,
    int Ntotal,
    int sub_n,
    int radix,
    int prim1,
    int batch,
    int inverse
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * sub_n;

    if (idx >= total) return;

    int b = idx / sub_n;
    int rem = idx % sub_n;

    int r0 = rem / prim1;
    int c  = rem % prim1;

    int input_base = b * sub_n;

    cd sum = make_cuDoubleComplex(0.0, 0.0);

    for (int m = 0; m < radix; ++m) {
        cd x = input[input_base + c + m * prim1];

        int idx1 = (r0 * m * (Ntotal / radix)) % Ntotal;
        cd w1 = get_twiddle(W, idx1, Ntotal, inverse);

        sum = cadd(sum, cmul(x, w1));
    }

    int idx2 = (r0 * c * (Ntotal / sub_n)) % Ntotal;
    cd tw = get_twiddle(W, idx2, Ntotal, inverse);

    sum = cmul(sum, tw);

    output[(b * radix + r0) * prim1 + c] = sum;
}




__global__
void final_dft_shared_kernel(
    const cd* __restrict__ input,
    cd* __restrict__ output,
    const cd* __restrict__ W,
    int Ntotal,
    int last_radix,
    int batch,
    int inverse
) {
    extern __shared__ cd sdata[];

    int b = blockIdx.x;
    int tid = threadIdx.x;

    if (b >= batch) return;
    const cd* row_in = input + b * last_radix;

    for (int m = tid; m < last_radix; m += blockDim.x) {
        sdata[m] = row_in[m];
    }

    __syncthreads();

    for (int k = tid; k < last_radix; k += blockDim.x) {
        cd sum = make_cuDoubleComplex(0.0, 0.0);

        for (int m = 0; m < last_radix; ++m) {
            int idx1 = (k * m * (Ntotal / last_radix)) % Ntotal;
            cd w = get_twiddle(W, idx1, Ntotal, inverse);
            sum = cadd(sum, cmul(sdata[m], w));
        }

        output[b * last_radix + k] = sum;
    }
}


__global__
void final_dft_global_kernel(
    const cd* __restrict__ input,
    cd* __restrict__ output,
    const cd* __restrict__ W,
    int Ntotal,
    int last_radix,
    int batch,
    int inverse
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * last_radix;

    if (idx >= total) return;

    int b = idx / last_radix;
    int k = idx % last_radix;

    cd sum = make_cuDoubleComplex(0.0, 0.0);

    for (int m = 0; m < last_radix; ++m) {
        cd x = input[b * last_radix + m];

        int idx1 = (k * m * (Ntotal / last_radix)) % Ntotal;
        cd w = get_twiddle(W, idx1, Ntotal, inverse);

        sum = cadd(sum, cmul(x, w));
    }

    output[idx] = sum;
}


__global__
void final_permute_kernel(
    const cd* __restrict__ input,
    cd* __restrict__ output,
    const int* __restrict__ factors,
    const int* __restrict__ suffix,
    const int* __restrict__ prefix,
    int num_factors,
    int N
) {
    int row_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_index >= N) return;

    int natural_index = 0;

    for (int i = 0; i < num_factors; ++i) {
        int digit = (row_index / suffix[i]) % factors[i];
        natural_index += digit * prefix[i];
    }

    output[natural_index] = input[row_index];
}


__global__
__global__
void final_permute_batched_kernel(
    const cd* __restrict__ input,
    cd* __restrict__ output,
    const int* __restrict__ factors,
    const int* __restrict__ suffix,
    const int* __restrict__ prefix,
    int num_factors,
    int N,
    int user_batch,
    double scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = user_batch * N;

    if (idx >= total) return;

    int b = idx / N;
    int row_index = idx % N;

    int natural_index = 0;

    for (int i = 0; i < num_factors; ++i) {
        int digit = (row_index / suffix[i]) % factors[i];
        natural_index += digit * prefix[i];
    }

    cd v = input[b * N + row_index];

    v = make_cuDoubleComplex(
        cuCreal(v) * scale,
        cuCimag(v) * scale
    );

    output[b * N + natural_index] = v;
}





class MixedRadixFFT_GPU {
public:
    int max_batch_ = 1024;
    explicit MixedRadixFFT_GPU(std::vector<int> factors, int max_batch)
    : factors_(std::move(factors)), max_batch_(max_batch)
    {
        if (factors_.empty()) {
            throw std::invalid_argument("Factor list must not be empty.");
        }
        
         max_batch_ = max_batch;
        
        if (max_batch_ <= 0) {
            throw std::invalid_argument("max_batch must be positive.");
         }


        N_ = 1;
        for (int f : factors_) {
            if (f <= 0) {
                throw std::invalid_argument("Factors must be positive.");
            }

            if (N_ > std::numeric_limits<int>::max() / f) {
                throw std::runtime_error("FFT size too large for this int-based implementation.");
            }

            N_ *= f;
        }

        build_permutation_tables();

   
        CHECK_CUDA(cudaMalloc(&d_a_, sizeof(cd) * N_ * max_batch_));
        CHECK_CUDA(cudaMalloc(&d_b_, sizeof(cd) * N_ * max_batch_));

        CHECK_CUDA(cudaMalloc(&d_twiddles_, sizeof(cd) * N_));

        CHECK_CUDA(cudaMalloc(&d_factors_, sizeof(int) * factors_.size()));
        CHECK_CUDA(cudaMalloc(&d_suffix_,  sizeof(int) * factors_.size()));
        CHECK_CUDA(cudaMalloc(&d_prefix_,  sizeof(int) * factors_.size()));

        CHECK_CUDA(cudaMemcpy(
            d_factors_,
            factors_.data(),
            sizeof(int) * factors_.size(),
            cudaMemcpyHostToDevice
        ));

        CHECK_CUDA(cudaMemcpy(
            d_suffix_,
            suffix_.data(),
            sizeof(int) * suffix_.size(),
            cudaMemcpyHostToDevice
        ));

        CHECK_CUDA(cudaMemcpy(
            d_prefix_,
            prefix_.data(),
            sizeof(int) * prefix_.size(),
            cudaMemcpyHostToDevice
        ));

        int threads = 256;
        int blocks = (N_ + threads - 1) / threads;

        init_twiddles<<<blocks, threads>>>(d_twiddles_, N_);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    ~MixedRadixFFT_GPU() {
        cudaFree(d_a_);
        cudaFree(d_b_);
        cudaFree(d_twiddles_);
        cudaFree(d_factors_);
        cudaFree(d_suffix_);
        cudaFree(d_prefix_);
    }

    std::vector<std::complex<double>>
    transform(const std::vector<std::complex<double>>& x,int inverse )
    {
        if (static_cast<int>(x.size()) != N_) {
            throw std::invalid_argument("Input size does not match product of factors.");
        }

        std::vector<cd> h_input(N_);

        for (int i = 0; i < N_; ++i) {
            h_input[i] = make_cuDoubleComplex(x[i].real(), x[i].imag());
        }

        CHECK_CUDA(cudaMemcpy(
            d_a_,
            h_input.data(),
            sizeof(cd) * N_,
            cudaMemcpyHostToDevice
        ));

        cd* in = d_a_;
        cd* out = d_b_;

        int batch = 1;
        int sub_n = N_;

        // ------------------------------------------------------------
        // All recursive split levels except the final direct DFT level.
        // ------------------------------------------------------------
        for (int stage = 0; stage < static_cast<int>(factors_.size()) - 1; ++stage) {
            int radix = factors_[stage];

            if (sub_n % radix != 0) {
                throw std::runtime_error("Invalid factorization.");
            }

            int prim1 = sub_n / radix;

            launch_split_stage(in, out, sub_n, radix, prim1, batch, inverse);

            std::swap(in, out);

            batch *= radix;
            sub_n = prim1;
        }

        // ------------------------------------------------------------
        // Final direct DFT on each row.
        // ------------------------------------------------------------
        int last_radix = factors_.back();

        if (sub_n != last_radix) {
            throw std::runtime_error("Internal factorization error.");
        }

        launch_final_dft(in, out, last_radix, batch, inverse);

        std::swap(in, out);

        // ------------------------------------------------------------
        // Final permutation to match CPU output order.
        // ------------------------------------------------------------
        int threads = 256;
        int blocks = (N_ + threads - 1) / threads;

        final_permute_kernel<<<blocks, threads>>>(
            in,
            out,
            d_factors_,
            d_suffix_,
            d_prefix_,
            static_cast<int>(factors_.size()),
            N_
        );

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        std::vector<cd> h_output(N_);

        CHECK_CUDA(cudaMemcpy(
            h_output.data(),
            out,
            sizeof(cd) * N_,
            cudaMemcpyDeviceToHost
        ));

        std::vector<std::complex<double>> result(N_);

        for (int i = 0; i < N_; ++i) {
            result[i] = std::complex<double>(
                cuCreal(h_output[i]),
                cuCimag(h_output[i])
            );
        }

        return result;
    }

void launch_final_dft_batched(
    const cd* in,
    cd* out,
    int last_radix,
    int batch,
    cudaStream_t stream,
    int inverse 
) {
    if (last_radix <= 1024) {
        int threads = std::min(1024, next_power_of_two(last_radix));
        int blocks = batch;
        size_t shared_bytes = sizeof(cd) * last_radix;

        final_dft_shared_kernel<<<blocks, threads, shared_bytes, stream>>>(
            in,
            out,
            d_twiddles_,
            N_,
            last_radix,
            batch,
            inverse
        );
    } else {
        int threads = 256;
        int total = batch * last_radix;
        int blocks = (total + threads - 1) / threads;

        final_dft_global_kernel<<<blocks, threads, 0, stream>>>(
            in,
            out,
            d_twiddles_,
            N_,
            last_radix,
            batch,
            inverse
        );
    }

    CHECK_CUDA(cudaGetLastError());
}

void transform_batch_device_impl(
    const cd* d_input,
    cd* d_output,
    int user_batch,
    int inverse,
    cudaStream_t stream = 0
) {
    if (user_batch <= 0) {
        throw std::invalid_argument("Batch size must be positive.");
    }

    if (user_batch > max_batch_) {
        throw std::invalid_argument("user_batch exceeds max_batch allocated in constructor.");
    }

    int total = user_batch * N_;

    cd* in = d_a_;
    cd* out = d_b_;

    CHECK_CUDA(cudaMemcpyAsync(
        in,
        d_input,
        sizeof(cd) * total,
        cudaMemcpyDeviceToDevice,
        stream
    ));

    int batch = user_batch;
    int sub_n = N_;

    for (int stage = 0; stage < static_cast<int>(factors_.size()) - 1; ++stage) {
        int radix = factors_[stage];

        if (sub_n % radix != 0) {
            throw std::runtime_error("Invalid factorization.");
        }

        int prim1 = sub_n / radix;

        launch_split_stage_batched(
            in,
            out,
            sub_n,
            radix,
            prim1,
            batch,
            inverse,
            stream
        );

        std::swap(in, out);

        batch *= radix;
        sub_n = prim1;
    }

    int last_radix = factors_.back();

    if (sub_n != last_radix) {
        throw std::runtime_error("Internal factorization error.");
    }

    launch_final_dft_batched(
        in,
        out,
        last_radix,
        batch,
        stream,
        inverse
    );

    std::swap(in, out);

    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    double scale = inverse ? (1.0 / static_cast<double>(N_)) : 1.0;

    final_permute_batched_kernel<<<blocks, threads, 0, stream>>>(
        in,
        out,
        d_factors_,
        d_suffix_,
        d_prefix_,
        static_cast<int>(factors_.size()),
        N_,
        user_batch,
        scale
    );

    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpyAsync(
        d_output,
        out,
        sizeof(cd) * total,
        cudaMemcpyDeviceToDevice,
        stream
    ));
}

void launch_split_stage_batched(
    const cd* in,
    cd* out,
    int sub_n,
    int radix,
    int prim1,
    int batch,
    int inverse,
    cudaStream_t stream
) {
    if (radix <= 1024) {
        int threads = std::min(1024, next_power_of_two(radix));
        int blocks = batch * prim1;
        size_t shared_bytes = sizeof(cd) * radix;

        split_stage_shared_kernel<<<blocks, threads, shared_bytes, stream>>>(
            in,
            out,
            d_twiddles_,
            N_,
            sub_n,
            radix,
            prim1,
            batch,
            inverse
        );
    } else {
        int threads = 256;
        int total = batch * sub_n;
        int blocks = (total + threads - 1) / threads;

        split_stage_global_kernel<<<blocks, threads, 0, stream>>>(
            in,
            out,
            d_twiddles_,
            N_,
            sub_n,
            radix,
            prim1,
            batch,
            inverse
        );
    }

    CHECK_CUDA(cudaGetLastError());
}

private:
    std::vector<int> factors_;
    std::vector<int> suffix_;
    std::vector<int> prefix_;

    int N_ = 1;

    cd* d_a_ = nullptr;
    cd* d_b_ = nullptr;
    cd* d_twiddles_ = nullptr;

    int* d_factors_ = nullptr;
    int* d_suffix_ = nullptr;
    int* d_prefix_ = nullptr;

private:
    static int next_power_of_two(int x) {
        int p = 1;
        while (p < x) p <<= 1;
        return p;
    }

    void build_permutation_tables() {
        int L = static_cast<int>(factors_.size());

        suffix_.resize(L);
        prefix_.resize(L);

        for (int i = 0; i < L; ++i) {
            int s = 1;
            for (int j = i + 1; j < L; ++j) {
                s *= factors_[j];
            }
            suffix_[i] = s;
        }

        for (int i = 0; i < L; ++i) {
            int p = 1;
            for (int j = 0; j < i; ++j) {
                p *= factors_[j];
            }
            prefix_[i] = p;
        }
    }

    void launch_split_stage(
        const cd* in,
        cd* out,
        int sub_n,
        int radix,
        int prim1,
        int batch,
        int inverse
    ) {
        if (radix <= 1024) {
            int threads = std::min(1024, next_power_of_two(radix));
            int blocks = batch * prim1;
            size_t shared_bytes = sizeof(cd) * radix;

            split_stage_shared_kernel<<<blocks, threads, shared_bytes>>>(
                in,
                out,
                d_twiddles_,
                N_,
                sub_n,
                radix,
                prim1,
                batch,
                inverse
            );
        } else {
            int threads = 256;
            int total = batch * sub_n;
            int blocks = (total + threads - 1) / threads;

            split_stage_global_kernel<<<blocks, threads>>>(
                in,
                out,
                d_twiddles_,
                N_,
                sub_n,
                radix,
                prim1,
                batch,
                inverse
            );
        }

        CHECK_CUDA(cudaGetLastError());
    }

    void launch_final_dft(
        const cd* in,
        cd* out,
        int last_radix,
        int batch,
        int inverse

    ) {
        if (last_radix <= 1024) {
            int threads = std::min(1024, next_power_of_two(last_radix));
            int blocks = batch;
            size_t shared_bytes = sizeof(cd) * last_radix;

            final_dft_shared_kernel<<<blocks, threads, shared_bytes>>>(
                in,
                out,
                d_twiddles_,
                N_,
                last_radix,
                batch,
                inverse
            );
        } else {
            int threads = 256;
            int total = batch * last_radix;
            int blocks = (total + threads - 1) / threads;

            final_dft_global_kernel<<<blocks, threads>>>(
                in,
                out,
                d_twiddles_,
                N_,
                last_radix,
                batch,
                inverse       
            );
        }

        CHECK_CUDA(cudaGetLastError());
    }
};


/*int main() {
    std::vector<int> factors = {3, 4};

    int N = 1;
    for (int f : factors) {
        N *= f;
    }

    int B = 4;

    MixedRadixFFT_GPU fft(factors, B);

    std::vector<cd> h_input(B * N);

    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            h_input[b * N + i] =
                make_cuDoubleComplex(i + 1 + 100 * b, 0.0);
        }
    }

    cd* d_input = nullptr;
    cd* d_freq = nullptr;
    cd* d_recovered = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, sizeof(cd) * B * N));
    CHECK_CUDA(cudaMalloc(&d_freq, sizeof(cd) * B * N));
    CHECK_CUDA(cudaMalloc(&d_recovered, sizeof(cd) * B * N));

    CHECK_CUDA(cudaMemcpy(
        d_input,
        h_input.data(),
        sizeof(cd) * B * N,
        cudaMemcpyHostToDevice
    ));

    fft.transform_batch_device_impl(d_input, d_freq,B,0,0);
    fft.transform_batch_device_impl(d_freq,d_recovered,B,1,0);

    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<cd> h_recovered(B * N);

    CHECK_CUDA(cudaMemcpy(
        h_recovered.data(),
        d_recovered,
        sizeof(cd) * B * N,
        cudaMemcpyDeviceToHost
    ));

    for (int b = 0; b < B; ++b) {
        std::cout << "\nBatch " << b << " recovered:\n";

        for (int i = 0; i < N; ++i) {
            cd v = h_recovered[b * N + i];

            std::cout << i << ": ("
                      << cuCreal(v) << ", "
                      << cuCimag(v) << ")\n";
        }
    }

    cudaFree(d_input);
    cudaFree(d_freq);
    cudaFree(d_recovered);

    return 0;
}
*/

