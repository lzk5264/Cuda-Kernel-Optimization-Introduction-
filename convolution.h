#pragma once
const int K = 257;
const int R = (K - 1) / 2;
const int BLOCK_DIM_X = 16;
const int BLOCK_DIM_Y = 16;
const int BLOCK_DIM = 16;
void convolution_cpu(const float *input, float *output, const int width, const int height, const float *kernel);
__global__ void transposeShared(const float* idata, float* odata, int width, int height);

__global__ void convolution_gpu_basic(float *input, float *output, int width, int height, float *kernel);

__global__ void convolution_gpu_row(const float* __restrict__  input,
                                    float* __restrict__        output,
                                    int width, int height,
                                    const float* __restrict__  kernel);

__global__ void convolution_gpu_row_shared(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width,
    int height,
    const float* __restrict__ kernel);

__global__ void convolution_gpu_row_shared_sp(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width,
    int height,
    const float* __restrict__ kernel);

__global__ void convolution_gpu_row_vec4(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width,
    int height,
    const float* __restrict__ kernel);
