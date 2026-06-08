import torch
import fft_cuda

# -------------------------
# init FFT
# -------------------------
factors = [3,5,11,17]
N = 2805

fft_cuda.init_fft(N, factors,64001)

# -------------------------
# test forward/inverse
# -------------------------
B = 23

x = torch.randn(B, N, dtype=torch.cdouble, device='cuda', requires_grad=True)

y = fft_cuda.fft(x)
z = fft_cuda.ifft(y)

print("Forward/IFFT error:", (z - x).abs().max().item())

# -------------------------
# test gradient
# -------------------------
loss = (y.abs()**2).sum()
loss.backward()

print("Gradient OK:", x.grad is not None)
print("Gradient sample:", x.grad[0, :5])

