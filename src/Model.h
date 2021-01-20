//
// Created by Armando Herrera on 1/19/21.
//

#ifndef SUPERRESOLUTION_MODEL_H
#define SUPERRESOLUTION_MODEL_H

#include <filesystem>
#include <torch/script.h>
#include <opencv2/core/mat.hpp>

class Model {
    torch::jit::script::Module module;
    size_t input_dim;
    size_t output_dim;
    torch::Device device;

public:
    Model(const std::filesystem::path &model_path, size_t upscale, size_t output_dims_size);

    cv::Mat run(const cv::Mat &input);
};


#endif //SUPERRESOLUTION_MODEL_H
