#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <assert.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cuda_runtime.h>
#include <stdio.h>

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
        float cos_val = freqs_cos[idx % dim];
        float sin_val = freqs_sin[idx % dim];

        // Apply rotation using real numbers for xq
        xq_out_r[idx] = xq_r[idx] * cos_val - xq_i[idx] * sin_val;
        xq_out_i[idx] = xq_r[idx] * sin_val + xq_i[idx] * cos_val;

        // Apply rotation using real numbers for xk
        xk_out_r[idx] = xk_r[idx] * cos_val - xk_i[idx] * sin_val;
        xk_out_i[idx] = xk_r[idx] * sin_val + xk_i[idx] * cos_val;
    }
}

void reshape_complex(float* inp, int B, int T, int C, float* out_real, float* out_imag) {
    #pragma unroll
    for (int b = 0; b < B; b++) {
        #pragma unroll 
        for (int t = 0; t < T; t++) {
            #pragma unroll
            for (int c = 0; c < C/2; c++) {
                int idx = b * T * C + t * C + 2 * c;
                out_real[b * T * (C/2) + t * (C/2) + c] = inp[idx];
                out_imag[b * T * (C/2) + t * (C/2) + c] = inp[idx + 1];
            }
        }
    }
}

extern "C" void apply_rotary_emb(
    float* xq_inp,
    float* xk_inp,
    float* freqs_cos,
    float* freqs_sin,
    float* xq_out,
    float* xk_out,
    int B,
    int T,
    int C
) {
    int dim_real = C / 2;
    int total_size = B * T * dim_real;

    // Allocate host memory for reshaped parts
    float* xq_real = (float*)malloc(total_size * sizeof(float));
    float* xq_imag = (float*)malloc(total_size * sizeof(float));
    float* xk_real = (float*)malloc(total_size * sizeof(float));
    float* xk_imag = (float*)malloc(total_size * sizeof(float));

    // Reshape xq and xk to real and imaginary parts on host
    reshape_complex(xq_inp, B, T, C, xq_real, xq_imag);
    reshape_complex(xk_inp, B, T, C, xk_real, xk_imag);

    float *d_xq_r, *d_xq_i, *d_xk_r, *d_xk_i;
    cudaMalloc(&d_xq_r, total_size * sizeof(float));
    cudaMalloc(&d_xq_i, total_size * sizeof(float));
    cudaMalloc(&d_xk_r, total_size * sizeof(float));
    cudaMalloc(&d_xk_i, total_size * sizeof(float));

    // Copy reshaped data to device
    cudaMemcpy(d_xq_r, xq_real, total_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_xq_i, xq_imag, total_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_xk_r, xk_real, total_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_xk_i, xk_imag, total_size * sizeof(float), cudaMemcpyHostToDevice);

    float *d_freqs_cos, *d_freqs_sin;
    cudaMalloc(&d_freqs_cos, dim_real * sizeof(float));
    cudaMalloc(&d_freqs_sin, dim_real * sizeof(float));
    cudaMemcpy(d_freqs_cos, freqs_cos, dim_real * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_freqs_sin, freqs_sin, dim_real * sizeof(float), cudaMemcpyHostToDevice);

    float *d_xq_out_r, *d_xq_out_i, *d_xk_out_r, *d_xk_out_i;
    cudaMalloc(&d_xq_out_r, total_size * sizeof(float));
    cudaMalloc(&d_xq_out_i, total_size * sizeof(float));
    cudaMalloc(&d_xk_out_r, total_size * sizeof(float));
    cudaMalloc(&d_xk_out_i, total_size * sizeof(float));

    int blockSize = 256;
    int numBlocks = (total_size + blockSize - 1) / blockSize;

    apply_rotary_emb_kernel<<<numBlocks, blockSize>>>(
        d_xq_r, d_xq_i, d_xk_r, d_xk_i,
        d_freqs_cos, d_freqs_sin,
        d_xq_out_r, d_xq_out_i,
        d_xk_out_r, d_xk_out_i,
        B, T, dim_real);

    // Combine real and imaginary parts and flatten
    float *xq_out_r_h = (float*)malloc(total_size * sizeof(float));
    float *xq_out_i_h = (float*)malloc(total_size * sizeof(float));
    float *xk_out_r_h = (float*)malloc(total_size * sizeof(float));
    float *xk_out_i_h = (float*)malloc(total_size * sizeof(float));

    cudaMemcpy(xq_out_r_h, d_xq_out_r, total_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(xq_out_i_h, d_xq_out_i, total_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(xk_out_r_h, d_xk_out_r, total_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(xk_out_i_h, d_xk_out_i, total_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Combine the real and imaginary parts into the output
    #pragma unroll
    for (int b = 0; b < B; b++) {
        #pragma unroll
        for (int t = 0; t < T; t++) {
            #pragma unroll
            for (int c = 0; c < dim_real; c++) {
                int idx_out = b * T * C + t * C + 2 * c;
                int idx_in = b * T * dim_real + t * dim_real + c;
                xq_out[idx_out] = xq_out_r_h[idx_in];
                xq_out[idx_out + 1] = xq_out_i_h[idx_in];
                xk_out[idx_out] = xk_out_r_h[idx_in];
                xk_out[idx_out + 1] = xk_out_i_h[idx_in];
            }
        }
    }

    // Free host memory
    free(xq_real);
    free(xq_imag);
    free(xk_real);
    free(xk_imag);
    free(xq_out_r_h);
    free(xq_out_i_h);
    free(xk_out_r_h);
    free(xk_out_i_h);

    // Free device memory
    cudaFree(d_xq_r);
    cudaFree(d_xq_i);
    cudaFree(d_xk_r);
    cudaFree(d_xk_i);
    cudaFree(d_freqs_cos);
    cudaFree(d_freqs_sin);
    cudaFree(d_xq_out_r);
    cudaFree(d_xq_out_i);
    cudaFree(d_xk_out_r);
    cudaFree(d_xk_out_i);
}

// Llama3 Rotary Positional Embedding CPU forward pass
void apply_rotary_emb_forward_cpu(
    float* xq_inp,
    float* xk_inp,
    float* freqs_cos,
    float* freqs_sin,
    float* xq_out,
    float* xk_out,
    int B,
    int T,
    int C
) {
    float* xq_real = (float*)malloc(B * T * (C/2) * sizeof(float));
    float* xq_imag = (float*)malloc(B * T * (C/2) * sizeof(float));
    float* xk_real = (float*)malloc(B * T * (C/2) * sizeof(float));
    float* xk_imag = (float*)malloc(B * T * (C/2) * sizeof(float));

    reshape_complex(xq_inp, B, T, C, xq_real, xq_imag);
    reshape_complex(xk_inp, B, T, C, xk_real, xk_imag);

    #pragma unroll
    for (int b = 0; b < B; b++) {
        #pragma unroll
        for (int t = 0; t < T; t++) {
            #pragma unroll
            for (int c = 0; c < C/2; c++) {
                int idx = b * T * (C/2) + t * (C/2) + c;
                float xq_r_val = xq_real[idx];
                float xq_i_val = xq_imag[idx];
                float xk_r_val = xk_real[idx];
                float xk_i_val = xk_imag[idx];

                float cos_val = freqs_cos[c];
                float sin_val = freqs_sin[c];

                xq_out[idx * 2] = xq_r_val * cos_val - xq_i_val * sin_val;
                xq_out[idx * 2 + 1] = xq_r_val * sin_val + xq_i_val * cos_val;

                xk_out[idx * 2] = xk_r_val * cos_val - xk_i_val * sin_val;
                xk_out[idx * 2 + 1] = xk_r_val * sin_val + xk_i_val * cos_val;
            }
        }
    }

    free(xq_real);
    free(xq_imag);
    free(xk_real);
    free(xk_imag);
}

int main() {
    int B = 2;
    int T = 2;
    int C = 2;

    printf("%s\n", "hello...");

    float* xq_inp = (float*)malloc(B * T * C * sizeof(float));
    float* xk_inp = (float*)malloc(B * T * C * sizeof(float));
    float* freqs_cos = (float*)malloc((C/2) * sizeof(float));
    float* freqs_sin = (float*)malloc((C/2) * sizeof(float));
    float* xq_out = (float*)malloc(B * T * C * sizeof(float));
    float* xk_out = (float*)malloc(B * T * C * sizeof(float));

    for (int i = 0; i < B * T * C; i++) {
        xq_inp[i] = i + 1;
        xk_inp[i] = i + 1;
    }
    for (int i = 0; i < C/2; i++) {
        freqs_cos[i] = cos(i);
        freqs_sin[i] = sin(i);
    }

    apply_rotary_emb_forward_cpu(xq_inp, xk_inp, freqs_cos, freqs_sin, xq_out, xk_out, B, T, C);

    for (int i = 0; i < B * T * C; i++) {
        printf("andrei kernel: xq_out at index %d = %f\n", i+1, xq_out[i]);
        printf("andrei kernel: xk_out at index %d = %f\n", i+1, xk_out[i]);
    }

    xq_inp = (float*)malloc(B * T * C * sizeof(float));
    xk_inp = (float*)malloc(B * T * C * sizeof(float));
    freqs_cos = (float*)malloc((C/2) * sizeof(float));
    freqs_sin = (float*)malloc((C/2) * sizeof(float));
    xq_out = (float*)malloc(B * T * C * sizeof(float));
    xk_out = (float*)malloc(B * T * C * sizeof(float));

    for (int i = 0; i < B * T * C; i++) {
        xq_inp[i] = i + 1;
        xk_inp[i] = i + 1;
    }
    for (int i = 0; i < C/2; i++) {
        freqs_cos[i] = cos(i);
        freqs_sin[i] = sin(i);
    }

    apply_rotary_emb(xq_inp, xk_inp, freqs_cos, freqs_sin, xq_out, xk_out, B, T, C);

    for (int i = 0; i < B * T * C; i++) {
        printf("ken kernel: xq_out at index %d = %f\n", i+1, xq_out[i]);
        printf("ken kernel: xk_out at index %d = %f\n", i+1, xk_out[i]);
    }


    free(xq_inp);
    free(xk_inp);
    free(freqs_cos);
    free(freqs_sin);
    free(xq_out);
    free(xk_out);
    printf("%s\n", "finished running...");
    return 0;
}