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

at::Tensor Model::run(const at::Tensor &input) {
    return module.forward({input}).toTensor().squeeze();
}

at::Tensor Model::preprocess(const at::Tensor &input) const {
    int64_t width = input.size(1), height = input.size(2);
    at::Tensor blocked = input.permute({2, 0, 1});
    CHECK(width < input_dim) << "Input image width is larger than model's input\n";
    CHECK(height < input_dim) << "Input image height is larger than model's input\n";
    // Break up into blocks if input is larger than model input
    if (width > input_dim || height > input_dim) {
        blocked = input.unfold(1, input_dim, input_dim).unfold(2, input_dim, input_dim);
        blocked = blocked.reshape({3, -1, blocked.size(3), blocked.size(4)});
        blocked = blocked.permute({1, 0, 2, 3});
    } else
        blocked = blocked.unsqueeze(0);
    return blocked;
}

at::Tensor Model::postprocess(const at::Tensor &input) {
    return torch::clamp((input * 255) + 0.5, 0, 255).permute({1, 2, 0}).to(torch::Device("cpu"), torch::kUInt8);
}

cv::Mat Model::run(const cv::Mat &input) {
    int64_t width = input.rows, height = input.cols;
    at::Tensor input_t = torch::from_blob(input.data, {width, height, 3});
    input_t = preprocess(input_t)
}

//cv::Mat Model::run_block(const cv::Mat &input) {
//    int64_t width = input.rows, height = input.cols;
//    CHECK(width < input_dim) << "Input image width is larger than model's input\n";
//    CHECK(height < input_dim) << "Input image height is larger than model's input\n";
//    at::Tensor input_t = torch::from_blob(input.data, {1, width, height, 3});
//    input_t = preprocess(input_t);
//    at::Tensor output = run(input_t);
//    output = postprocess(output);
//    auto *output_ptr = output.data_ptr<uint8_t>();
//    return cv::Mat(cv::Size{static_cast<int>(width * 4), static_cast<int>(height * 4)}, CV_8UC3, output_ptr);
//}
