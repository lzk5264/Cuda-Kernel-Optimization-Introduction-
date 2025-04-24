#include <iostream>
#include <chrono>
#include <hip/hip_runtime.h>
#include "image_processing.h"
#include "convolution.h"

// 定义卷积模式枚举

// 执行CPU卷积并输出
void perform_cpu_convolution(float *h_input, float *h_output, int width, int height, int channels, int img_size, 
                            float *kernel)
{
    auto start = std::chrono::high_resolution_clock::now();
    convolution_cpu(h_input, h_output, width, height, kernel);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end - start;
    std::cout << "CPU convolution time: " << cpu_time.count() << " ms" << std::endl;

    gray_data2gray_image("./output_image/cpu_output.png", width, height, channels, h_output);
}

// 执行GPU基本卷积函数
void perform_gpu_basic_convolution(float *h_input, float *h_output, float *h_kernel,int width, int height, int channels)
{
    size_t img_size = width * height;
    float *d_input, *d_output, *d_kernel;
    hipMalloc((void**)&d_input, img_size * sizeof(float));
    hipMalloc((void**)&d_output, img_size * sizeof(float));
    hipMalloc((void**)&d_kernel, K * K * sizeof(float));

    hipMemcpy(d_input, h_input, img_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_kernel, h_kernel, K * K * sizeof(float), hipMemcpyHostToDevice);

    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    auto start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(convolution_gpu_basic, grid, block, 0, 0, d_input, d_output, width, height, d_kernel);
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_time = end - start;
    std::cout << "GPU basic convolution time: " << gpu_time.count() << " ms" << std::endl;

    hipMemcpy(h_output, d_output, img_size * sizeof(float), hipMemcpyDeviceToHost);
    hipFree(d_input);
    hipFree(d_output);
    hipFree(d_kernel);

    gray_data2gray_image("./output_image/gpu_output_basic.png", width, height, channels, h_output);
    std::cout << "GPU basic convolution output saved." << std::endl;
}

// 执行GPU分离卷积函数
void perform_gpu_separable_convolution(float *h_input, float *h_output, float *h_kernel, int width, int height, int channels)
{
    size_t img_size = width * height;
    float *d_input, *d_output, *d_kernel, *d_tmp;
    hipMalloc((void**)&d_input, img_size * sizeof(float));
    hipMalloc((void**)&d_output, img_size * sizeof(float));
    hipMalloc((void**)&d_kernel, K * sizeof(float));
    hipMalloc((void**)&d_tmp, img_size * sizeof(float));

    hipMemcpy(d_input, h_input, img_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_kernel, h_kernel, K * sizeof(float), hipMemcpyHostToDevice);

    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    dim3 grid_vec((width / 4 + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    auto start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(convolution_gpu_row_shared, grid, block, 0, 0, d_input, d_tmp, width, height, d_kernel);
    hipLaunchKernelGGL(transposeShared, grid, block, 0, 0, d_tmp, d_input, width, height);
    hipLaunchKernelGGL(convolution_gpu_row_shared, grid, block, 0, 0, d_input, d_tmp, height, width, d_kernel);
    hipLaunchKernelGGL(transposeShared, grid, block, 0, 0, d_tmp, d_output, height, width);
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_time = end - start;
    std::cout << "GPU separable convolution time: " << gpu_time.count() << " ms" << std::endl;

    hipMemcpy(h_output, d_output, img_size * sizeof(float), hipMemcpyDeviceToHost);
    hipFree(d_input);
    hipFree(d_output);
    hipFree(d_kernel);

    gray_data2gray_image("./output_image/gpu_output_separate.png", width, height, channels, h_output);
    std::cout << "GPU separable convolution output saved." << std::endl;
}

int main()
{
    // 读取图像
    int width, height, channels;
    unsigned char *img_data = stbi_load("./input_image/input.png", &width, &height, &channels, 0);
    if (!img_data) {
        std::cerr << "Error loading image: " << stbi_failure_reason() << std::endl;
        return -1;
    }

    size_t img_size = width * height;
    float *h_input = new float[img_size];
    float *h_kernel = new float[K * K];
    // 初始化卷积核
    for (int i = 0; i < K * K; ++i) {
        h_kernel[i] = 1.0f / (K * K); // 均值卷积核
    }
    
    // 转换为灰度图并归一化到0-1
    for (int i = 0; i < img_size; ++i) {
        unsigned char r = img_data[i * channels];
        unsigned char g = img_data[i * channels + 1];
        unsigned char b = img_data[i * channels + 2];
        h_input[i] = (0.299f*r + 0.587f*g + 0.114f*b) / 255.0f;
    }
    stbi_image_free(img_data);

    // 执行GPU卷积
    float *h_output = new float[img_size];
    perform_gpu_basic_convolution(h_input, h_output, h_kernel,width, height, channels);

    float *h_kernel_separable = new float[K];
    for (int i = 0; i < K; ++i) {
        h_kernel_separable[i] = 1.0f / K; // 均值卷积核
    }
    float *h_output_separable = new float[img_size];

    perform_gpu_separable_convolution(h_input, h_output_separable, h_kernel_separable,width, height, channels);

    if (!compare_image(h_output, h_output_separable, img_size)) {
        std::cerr << "GPU basic and separable convolution results do not match!" << std::endl;
    } else {
        std::cout << "GPU basic and separable convolution results match." << std::endl;
    }

    // 释放内存
    delete[] h_input;
    delete[] h_kernel;
    delete[] h_output;
    delete[] h_kernel_separable;
    delete[] h_output_separable;
    
    return 0;
}