#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <torch/extension.h>

// RMSNorm coop-group kernel
__global__ void rmsnorm_forward_kernel(
    float* __restrict__ out,
    const float* __restrict__ inp,
    const float* __restrict__ weight,
    int B,
    int T,
    int C
) {
    namespace cg = cooperative_groups;
    static constexpr unsigned WARP_SIZE = 32;
    static constexpr float eps = 1e-6f;

    int num_warps = blockDim.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int idx = blockIdx.x;

    __shared__ float shared[WARP_SIZE];
    const float *x = inp + idx * C;

    float thread_sum_of_squares = 0.0f;

    #pragma unroll
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float xi = x[i];
        thread_sum_of_squares += xi * xi;
    }

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> warp = cg::tiled_partition<WARP_SIZE>(block);

    float warp_sum_of_squares = cg::reduce(warp, thread_sum_of_squares, cg::plus<float>{}); // sum(x * x)
    if (lane_id == 0) {
        shared[warp_id] = warp_sum_of_squares;
        __syncthreads();
    }

    warp_sum_of_squares = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
    float block_sum_of_squares = cg::reduce(warp, warp_sum_of_squares, cg::plus<float>{}); // sum(x * x)

    // compute rms
    float rms_val = rsqrtf(block_sum_of_squares / C + eps);

    float *o = out + idx * C;

    #pragma unroll
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float n =  __ldcs(x+i) * rms_val;
        __stcs(o+i, n * weight[i]);
    }
}

// Binding
torch::Tensor rmsnorm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int B,
    int T,
    int C
) {
    auto output = torch::empty_like(input);

    const int block_size = 1024;
    const int grid_size = B * T;

    rmsnorm_forward_kernel<<<grid_size, block_size>>>(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        B,
        T,
        C
    );

    return output;
}
