import os
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.cpp_extension import load_inline

cuda_source_files = ['kernels/rmsnorm_forward.cu']
cuda_sources_list = [open(file).read() for file in cuda_source_files]
cuda_sources = '\n'.join(cuda_sources_list)

cpp_source = '''
torch::Tensor rmsnorm_forward(torch::Tensor input, torch::Tensor weight, int B, int T, int C);
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

    def forward(self, inp):
        B, T, C = inp.size()
        return rmsnorm_extension.rmsnorm_forward(inp, self.weight, B, T, C) 

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
    custom_rmsnorm = RMSNormCustom(input_tensor.shape[-1]).cuda()
    pytorch_rmsnorm = RMSNorm(input_tensor.shape[-1]).cuda()

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof_custom:
        custom_output = custom_rmsnorm(input_tensor)
        print(custom_output)

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof_pytorch:
        pytorch_output = pytorch_rmsnorm(input_tensor)
        print(pytorch_output)

    print("Custom RMSNorm Profiling:")
    print(prof_custom.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=10, header="Custom RMSNorm CUDA Profiling"))

    print("\nPyTorch LayerNorm Profiling:")
    print(prof_pytorch.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=10, header="PyTorch LayerNorm CUDA Profiling"))

