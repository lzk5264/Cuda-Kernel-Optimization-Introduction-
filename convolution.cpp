#include "convolution.h"

void convolution_cpu(const float *input, float *output, const int width, const int height, const float *kernel) {

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            for (int kx = 0; kx < K; ++kx) {
                for (int ky = 0; ky < K; ++ky) {
                    int ix = x + kx - R;
                    int iy = y + ky - R;
                    if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                        sum += input[iy * width + ix] * kernel[ky * K + kx];
                    }
                }
            }
            output[y * width + x] = sum;
        }
    }
}


__global__ void convolution_gpu_basic(float *input, float *output, int width, int height, float* kernel)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height)
        return;
    
    float sum = 0.0f;
    
    int y_start = max(0, y - R);
    int y_end = min(height - 1, y + R);
    int x_start = max(0, x - R);
    int x_end = min(width - 1, x + R);
    
    for (int iy = y_start; iy <= y_end; ++iy) {
        int ky = iy - y + R;
        for (int ix = x_start; ix <= x_end; ++ix) {
            int kx = ix - x + R;
            sum += input[iy * width + ix] * kernel[ky * K + kx];
        }
    }
    output[y * width + x] = sum;
}

__global__ void transposeShared(const float* idata, float* odata, int width, int height)
{
    __shared__ float tile[BLOCK_DIM][BLOCK_DIM + 1];  // +1 避免 shared memory bank conflict

    int x = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int y = blockIdx.y * BLOCK_DIM + threadIdx.y;

    if (x < width && y < height)
        tile[threadIdx.y][threadIdx.x] = idata[y * width + x];

    __syncthreads();

    x = blockIdx.y * BLOCK_DIM + threadIdx.x;
    y = blockIdx.x * BLOCK_DIM + threadIdx.y;

    if (x < height && y < width)
        odata[y * height + x] = tile[threadIdx.x][threadIdx.y];
}

__global__ void convolution_gpu_row(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width,
    int height,
    const float* __restrict__ kernel)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;
    float sum = 0.0f;
    for (int k = -R; k <= R; ++k) {
        int ix = x + k;
        sum += (ix >= 0 && ix < width) ? input[y * width + ix] * kernel[R + k] : 0.0f;
    }
    output[y * width + x] = sum;
}

__global__ void convolution_gpu_row_shared(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width,
    int height,
    const float* __restrict__ kernel)
{
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    const int bx = blockIdx.x * blockDim.x;
    if (gx >= width || gy >= height) return;

    __shared__ float kernel_shared[K];
    __shared__ float tile[BLOCK_DIM_Y * (BLOCK_DIM_X + 2 * R + 1)];
    
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        kernel_shared[k] = kernel[k];
    }
    // Load the tile into shared memory
    const int shift = threadIdx.y * (BLOCK_DIM_X + 2 * R);
   
    for (int tx = threadIdx.x; tx < BLOCK_DIM_X + 2 * R; tx += BLOCK_DIM_X)
    {
        int ix = bx + tx - R;
        int iy = gy;

        tile[shift + tx] = (ix >= 0 && ix < width) ? input[iy * width + ix] : 0.0f;
    }

    __syncthreads(); // 确保所有数据加载完成

    float sum = 0.0f;
    for (int k = -R; k <= R; ++k) {
            sum += tile[shift + threadIdx.x + k + R] * kernel_shared[R + k];
    }
    output[gy * width + gx] = sum;
}

__global__ void convolution_gpu_row_vec4(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width,
    int height,
    const float* __restrict__ kernel)
{
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;  // index in float4
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (gy >= height || gx * 4 >= width) return;

    float4 result = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (int k = -R; k <= R; ++k)
    {
        int xk0 = gx * 4 + 0 + k;
        int xk1 = gx * 4 + 1 + k;
        int xk2 = gx * 4 + 2 + k;
        int xk3 = gx * 4 + 3 + k;

        float coeff = kernel[k + R];

        float v0 = (xk0 >= 0 && xk0 < width) ? input[gy * width + xk0] : 0.0f;
        float v1 = (xk1 >= 0 && xk1 < width) ? input[gy * width + xk1] : 0.0f;
        float v2 = (xk2 >= 0 && xk2 < width) ? input[gy * width + xk2] : 0.0f;
        float v3 = (xk3 >= 0 && xk3 < width) ? input[gy * width + xk3] : 0.0f;

        result.x += v0 * coeff;
        result.y += v1 * coeff;
        result.z += v2 * coeff;
        result.w += v3 * coeff;
    }

    float* out_ptr = output + gy * width + gx * 4;
    out_ptr[0] = result.x;
    out_ptr[1] = result.y;
    out_ptr[2] = result.z;
    out_ptr[3] = result.w;
}

   
__global__ void convolution_gpu_row_shared_sp(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width,
    int height,
    const float* __restrict__ kernel)
{
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    const int tx = threadIdx.x;
    if (gx >= width || gy >= height) return;

    __shared__ float kernel_shared[K];
    __shared__ float tile[BLOCK_DIM_Y * (BLOCK_DIM_X + 2 * R)];
    
    if (tx < K) {
        kernel_shared[tx] = kernel[tx];
    }

    // Load the tile into shared memory
    const int shift = threadIdx.y * (BLOCK_DIM_X + 2 * R);
    int ix = gx - R;
    int iy = gy;
    tile[shift + tx] = (ix >= 0) ? input[iy * width + ix] : 0.0f;
    ix = gx + R;
    tile[shift + tx + 2 * R] = (ix < width) ? input[iy * width + ix] : 0.0f;

    __syncthreads(); // 确保所有数据加载完成

    float sum = 0.0f;
    for (int k = -R; k <= R; ++k) {
            sum += tile[shift + tx + k + R] * kernel_shared[R + k];
    }
    output[gy * width + gx] = sum;
}