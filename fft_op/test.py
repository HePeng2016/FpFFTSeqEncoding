import torch
import fft_cuda
import time 


# -------------------------
# init FFT
# -------------------------
factors = [3, 4]
N = 12

fft_cuda.init_fft(N, factors, 64)

# -------------------------
# test forward/inverse
# -------------------------
B = 2

x = torch.randn(B, N, dtype=torch.cdouble, device='cuda', requires_grad=True)

y = fft_cuda.fft(x)
z = fft_cuda.ifft(y)



factors = [11,17,13];
N =11*17*13;
fft_cuda.init_fft(N, factors, 64000);
B = 64000

x = torch.randn(B, N, dtype=torch.cdouble, device='cuda', requires_grad=True)

y = fft_cuda.fft(x)
z = fft_cuda.ifft(y)

start_time = time.perf_counter()
y = fft_cuda.fft(x)
z = fft_cuda.ifft(y)
end_time = time.perf_counter()
end_time - start_time

start_time = time.perf_counter()
y=torch.fft.fft(x);
z=torch.fft.ifft(y);
end_time = time.perf_counter()
end_time - start_time

torch.fft.fft()
torch.fft.ifft(input)




print("Forward/IFFT error:", (z - x).abs().max().item())

# -------------------------
# test gradient
# -------------------------
loss = (y.abs()**2).sum()
loss.backward()

print("Gradient OK:", x.grad is not None)
print("Gradient sample:", x.grad[0, :5])





