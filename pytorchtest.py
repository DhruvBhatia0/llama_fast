import os
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.cpp_extension import load_inline

cuda_source_files = ['kernels/rmsnorm_forward.cu']
cuda_sources_list = [open(file).read() for file in cuda_source_files]
cuda_sources = '\n'.join(cuda_sources_list)

cpp_source = '''
torch::Tensor rmsnorm_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int B, int T, int C);
'''

# Compile and load the CUDA extension
rmsnorm_extension = load_inline(
    name='rmsnorm_extension',
    cpp_sources=cpp_source,
    cuda_sources=cuda_sources,
    functions=['rmsnorm_forward'],
    with_cuda=True,
    extra_cflags=['-O2'],
)

class RMSNormCustom(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, inp):
        B, T, C = inp.size()
        # out = torch.empty_like(inp)
        # rms = torch.empty(B, T, device=inp.device, dtype=inp.dtype)

        return rmsnorm_extension.rmsnorm_forward(inp, self.weight, self.bias, B, T, C) 

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# profiling stuff
if __name__ == "__main__":
    input_tensor = torch.randn(8, 1024, 768, device='cuda')
    custom_rmsnorm = RMSNormCustom(768).cuda()
    pytorch_rmsnorm = RMSNorm(768).cuda()

    # Warm-up to avoid measuring initial setup overhead
    for _ in range(10):
        custom_output = custom_rmsnorm(input_tensor)
        pytorch_output = pytorch_rmsnorm(input_tensor)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof_custom:
        custom_output = custom_rmsnorm(input_tensor)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof_pytorch:
        pytorch_output = pytorch_rmsnorm(input_tensor)

    print("Custom RMSNorm Profiling:")
    print(prof_custom.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print("\nPyTorch LayerNorm Profiling:")
    print(prof_pytorch.key_averages().table(sort_by="cuda_time_total", row_limit=10))
