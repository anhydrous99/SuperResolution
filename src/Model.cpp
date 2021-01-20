//
// Created by Armando Herrera on 1/19/21.
//

#include "Model.h"

#include <glog/logging.h>

Model::Model(const std::filesystem::path &model_path, size_t upscale, size_t output_dims_size) : device("cpu") {
    try {
        module = torch::jit::load(model_path.string());
    } catch(const c10::Error& e) {
        LOG(FATAL) << "error loading module\n" << e.msg();
    }

    if (torch::hasCUDA()) {
        device = torch::Device(torch::kCUDA);
        module.to(device);
    }

    input_dim = output_dims_size / upscale;
    output_dim = output_dims_size;
}

cv::Mat Model::run(const cv::Mat &input) {
    at::Tensor input_t = torch::from_blob(input.data, {1, static_cast<long long>(input_dim), static_cast<long long>(input_dim), 3});
    input_t = input_t.permute({0, 3, 1, 2}).to(torch::kFloat32) / 255.f;
    input_t = input_t.to(device);
    at::Tensor output = module.forward({input_t}).toTensor().squeeze();
    output = torch::clamp((output * 255) + 0.5, 0, 255).permute({1, 2, 0}).to(torch::Device("cpu"), torch::kUInt8);
    auto *output_ptr = output.data_ptr<uint8_t>();
    return cv::Mat(cv::Size{static_cast<int>(output_dim), static_cast<int>(output_dim)}, CV_8UC3, output_ptr);
}
