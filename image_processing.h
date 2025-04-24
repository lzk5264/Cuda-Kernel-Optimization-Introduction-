#pragma once

#include <iostream>
#include <cmath>
// 移除实现宏，只包含头文件
#include <stb_image.h>
#include <stb_image_write.h>

bool compare_image(float *image1, float *image2, int size);
void gray_data2gray_image(const char *path, int width, int height, int channels, const float *data);

float* load_gray_image(const char *path, int &width, int &height, int &channels);