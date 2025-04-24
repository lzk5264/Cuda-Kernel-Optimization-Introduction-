#include "image_processing.h"

/**
 * @brief Compares two images for similarity.
 * 
 * @param image1 Pointer to the first image data.
 * @param image2 Pointer to the second image data.
 * @param size Size of the images.
 * @return true if the images are similar, false otherwise.
 */
bool compare_image(float *image1, float *image2, int size)
{
    for (int i = 0; i < size; ++i)
    {
        if (fabs(image1[i] - image2[i]) > 0.1)
        {
            return false;
        }
    }
    return true;
}
/**
 * @brief Converts grayscale data to a grayscale image file.
 * 
 * @param path Path to save the image.
 * @param width Width of the image.
 * @param height Height of the image.
 * @param channels Number of channels in the image.
 * @param data Pointer to the image data.
 */
void gray_data2gray_image(const char *path, int width, int height, int channels, const float *data)
{
    unsigned char *output_image = new unsigned char[width * height * channels];
    for (int i = 0; i < width * height; ++i)
    {
        for (int j = 0; j < channels; ++j)
        {
            output_image[i * channels + j] = static_cast<unsigned char>(data[i] * 255.0f);
        }
    }
    stbi_write_png(path, width, height, channels, output_image, width * channels);
    delete[] output_image;
}

float* load_gray_image(const char *path, int &width, int &height, int &channels)
{
    unsigned char *image = stbi_load(path, &width, &height, &channels, 0);
    if (!image)
    {
        std::cerr << "Error loading image: " << stbi_failure_reason() << std::endl;
        return nullptr;
    }
    size_t img_size = width * height;
    float *data = new float[img_size];
    for (int i = 0; i < img_size; ++i)
    {
        data[i] = static_cast<float>(image[i]) / 255.0f;
    }
    stbi_image_free(image);
    return data;
}