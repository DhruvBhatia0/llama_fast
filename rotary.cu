#include <cuda_runtime.h>
#include <stdio.h>

__global__ void apply_rotary_emb_kernel(
    float *xq_r, float *xq_i,
    float *xk_r, float *xk_i,
    float *freqs_cos, float *freqs_sin,
    float *xq_out_r, float *xq_out_i,
    float *xk_out_r, float *xk_out_i,
    int batch_size, int seq_len, int dim) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * seq_len * dim;

    if (idx < total_size) {
        float cos_val = freqs_cos[idx];
        float sin_val = freqs_sin[idx];

        // Apply rotation using real numbers for xq
        xq_out_r[idx] = xq_r[idx] * cos_val - xq_i[idx] * sin_val;
        xq_out_i[idx] = xq_r[idx] * sin_val + xq_i[idx] * cos_val;

        // Apply rotation using real numbers for xk
        xk_out_r[idx] = xk_r[idx] * cos_val - xk_i[idx] * sin_val;
        xk_out_i[idx] = xk_r[idx] * sin_val + xk_i[idx] * cos_val;
    }
}

extern "C" void apply_rotary_emb(
    float *xq, float *xk,
    float *freqs_cos, float *freqs_sin,
    float *xq_out, float *xk_out,
    int batch_size, int seq_len, int dim) {

    int total_size = batch_size * seq_len * dim;

    float *xq_r, *xq_i, *xk_r, *xk_i;
    cudaMalloc(&xq_r, total_size * sizeof(float));
    cudaMalloc(&xq_i, total_size * sizeof(float));
    cudaMalloc(&xk_r, total_size * sizeof(float));
    cudaMalloc(&xk_i, total_size * sizeof(float));

    // Reshape xq and xk to real and imaginary parts
    // (This part should be done on the host before copying to device memory)

    float *d_freqs_cos, *d_freqs_sin;
    cudaMalloc(&d_freqs_cos, total_size * sizeof(float));
    cudaMalloc(&d_freqs_sin, total_size * sizeof(float));
    cudaMemcpy(d_freqs_cos, freqs_cos, total_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_freqs_sin, freqs_sin, total_size * sizeof(float), cudaMemcpyHostToDevice);

    float *d_xq_out_r, *d_xq_out_i, *d_xk_out_r, *d_xk_out_i;
    cudaMalloc(&d_xq_out_r, total_size * sizeof(float));
    cudaMalloc(&d_xq_out_i, total_size * sizeof(float));
    cudaMalloc(&d_xk_out_r, total_size * sizeof(float));
    cudaMalloc(&d_xk_out_i, total_size * sizeof(float));

    int blockSize = 256;
    int numBlocks = (total_size + blockSize - 1) / blockSize;

    apply_rotary_emb_kernel<<<numBlocks, blockSize>>>(
        xq_r, xq_i, xk_r, xk_i,
        d_freqs_cos, d_freqs_sin,
        d_xq_out_r, d_xq_out_i,
        d_xk_out_r, d_xk_out_i,
        batch_size, seq_len, dim);

    // Combine real and imaginary parts and flatten
    // (This part should be done on the host after copying from device memory)

    cudaMemcpy(xq_out, d_xq_out_r, total_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(xk_out, d_xk_out_r, total_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(xq_r);
    cudaFree(xq_i);
    cudaFree(xk_r);
    cudaFree(xk_i);
    cudaFree(d_freqs_cos);
    cudaFree(d_freqs_sin);
    cudaFree(d_xq_out_r);
    cudaFree(d_xq_out_i);
    cudaFree(d_xk_out_r);
    cudaFree(d_xk_out_i);
}
