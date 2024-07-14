#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

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

__global__ void apply_rotary_emb_kernel(
    const float* xq, const float* xk,
    const float* freqs_cos, const float* freqs_sin,
    float* xq_out, float* xk_out,
    int B, int T, int C) {

    int b = blockIdx.x;
    int t = blockIdx.y;
    int c = threadIdx.x;

    if (b < B && t < T && c < C / 2) {
        int idx = b * T * C + t * C + 2 * c;
        float xq_r_val = xq[idx];
        float xq_i_val = xq[idx + 1];
        float xk_r_val = xk[idx];
        float xk_i_val = xk[idx + 1];

        float cos_val = freqs_cos[c];
        float sin_val = freqs_sin[c];

        xq_out[idx] = xq_r_val * cos_val - xq_i_val * sin_val;
        xq_out[idx + 1] = xq_r_val * sin_val + xq_i_val * cos_val;
        xk_out[idx] = xk_r_val * cos_val - xk_i_val * sin_val;
        xk_out[idx + 1] = xk_r_val * sin_val + xk_i_val * cos_val;
    }
}

void apply_rotary_emb_forward_cpu(
    const float* xq_inp,
    const float* xk_inp,
    const float* freqs_cos,
    const float* freqs_sin,
    float* xq_out,
    float* xk_out,
    int B,
    int T,
    int C
) {
    float* d_xq_inp;
    float* d_xk_inp;
    float* d_freqs_cos;
    float* d_freqs_sin;
    float* d_xq_out;
    float* d_xk_out;

    // Allocate device memory
    cudaMalloc((void**)&d_xq_inp, B * T * C * sizeof(float));
    cudaMalloc((void**)&d_xk_inp, B * T * C * sizeof(float));
    cudaMalloc((void**)&d_freqs_cos, (C / 2) * sizeof(float));
    cudaMalloc((void**)&d_freqs_sin, (C / 2) * sizeof(float));
    cudaMalloc((void**)&d_xq_out, B * T * C * sizeof(float));
    cudaMalloc((void**)&d_xk_out, B * T * C * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_xq_inp, xq_inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_xk_inp, xk_inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_freqs_cos, freqs_cos, (C / 2) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_freqs_sin, freqs_sin, (C / 2) * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 grid(B, T);
    dim3 block(C / 2);

    // Launch kernel
    apply_rotary_emb_kernel<<<grid, block>>>(
        d_xq_inp, d_xk_inp, d_freqs_cos, d_freqs_sin, d_xq_out, d_xk_out, B, T, C);

    // Copy output data back to host
    cudaMemcpy(xq_out, d_xq_out, B * T * C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(xk_out, d_xk_out, B * T * C * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_xq_inp);
    cudaFree(d_xk_inp);
    cudaFree(d_freqs_cos);
    cudaFree(d_freqs_sin);
    cudaFree(d_xq_out);
    cudaFree(d_xk_out);
}

// Original function definition for CPU comparison
void apply_rotary_emb_forward_cpu_original(
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

void print_tensor(float* tensor, int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < C; c++) {
                printf("%f ", tensor[b * T * C + t * C + c]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

int main() {
    int B = 64; // Batch size
    int T = 128; // Sequence length
    int C = 256; // Embedding dimension
    int iterations = 100; // Number of iterations for averaging

    // Allocate memory for inputs and outputs
    float* xq_inp = (float*)malloc(B * T * C * sizeof(float));
    float* xk_inp = (float*)malloc(B * T * C * sizeof(float));
    float* freqs_cos = (float*)malloc(C/2 * sizeof(float));
    float* freqs_sin = (float*)malloc(C/2 * sizeof(float));
    float* xq_out_original = (float*)malloc(B * T * C * sizeof(float));
    float* xk_out_original = (float*)malloc(B * T * C * sizeof(float));
    float* xq_out_cuda = (float*)malloc(B * T * C * sizeof(float));
    float* xk_out_cuda = (float*)malloc(B * T * C * sizeof(float));

    // Initialize inputs with some values
    for (int i = 0; i < B * T * C; i++) {
        xq_inp[i] = rand() % 10;
        xk_inp[i] = rand() % 10;
    }
    for (int i = 0; i < C/2; i++) {
        freqs_cos[i] = rand() % 10;
        freqs_sin[i] = rand() % 10;
    }

    // Warm-up runs for both versions
    apply_rotary_emb_forward_cpu_original(xq_inp, xk_inp, freqs_cos, freqs_sin, xq_out_original, xk_out_original, B, T, C);
    apply_rotary_emb_forward_cpu(xq_inp, xk_inp, freqs_cos, freqs_sin, xq_out_cuda, xk_out_cuda, B, T, C);

    // Measure CPU performance
    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        apply_rotary_emb_forward_cpu_original(xq_inp, xk_inp, freqs_cos, freqs_sin, xq_out_original, xk_out_original, B, T, C);
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_cpu = end_cpu - start_cpu;
    double avg_duration_cpu = duration_cpu.count() / iterations;
    printf("CPU execution time: %f seconds\n", avg_duration_cpu);

    // Measure CUDA performance
    auto start_cuda = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        apply_rotary_emb_forward_cpu(xq_inp, xk_inp, freqs_cos, freqs_sin, xq_out_cuda, xk_out_cuda, B, T, C);
    }
    auto end_cuda = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_cuda = end_cuda - start_cuda;
    double avg_duration_cuda = duration_cuda.count() / iterations;
    printf("CUDA execution time: %f seconds\n", avg_duration_cuda);

    // Free allocated memory
    free(xq_inp);
    free(xk_inp);
    free(freqs_cos);
    free(freqs_sin);
    free(xq_out_original);
    free(xk_out_original);
    free(xq_out_cuda);
    free(xk_out_cuda);

    return 0;
}
