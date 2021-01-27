//
// Created by Armando Herrera on 1/19/21.
// TODO: Create model with dynamic batch size, this will accelerate super-sampling for large images or videos on CUDA
//

#include "Model.h"
#include "utils.h"

#include <glog/logging.h>
#include <opencv2/imgproc.hpp>
#include <algorithm>


namespace idx = torch::indexing;

Model::Model(const std::filesystem::path &model_path, int64_t upscale, int64_t output_size, int64_t prefetch_size)
        : device("cpu"), prefetch_size(prefetch_size) {
    try {
        module = torch::jit::load(model_path.string());
    } catch (const c10::Error &e) {
        LOG(FATAL) << "error loading module\n" << e.msg();
    }
    module.eval();

    if (torch::hasCUDA()) {
        device = torch::Device(torch::kCUDA);
        module.to(device);
    }

    input_dim = output_size / upscale;
    output_dim = output_size;
    scale = upscale;
}

std::vector<at::Tensor> Model::run(const std::vector<at::Tensor> &input) {
    std::vector<at::Tensor> output;
    for (uint64_t i = 0; i < input.size(); i += prefetch_size) {
        std::vector<at::Tensor> device_tensor;
        for (auto j = i; j < std::min(i + prefetch_size, input.size()); j++)
            device_tensor.push_back(input[j].to(device));
        for (auto &tensor : device_tensor)
            tensor = module.forward({tensor}).toTensor();
        for (const auto &tensor : device_tensor)
            output.push_back(tensor.to(torch::Device("cpu")));
    }
    return output;
}

std::vector<at::Tensor> Model::preprocess(const at::Tensor &input) const {
    int64_t height = input.size(0), width = input.size(1);
    std::vector<at::Tensor> output;
    for (int64_t i = 0; i < height; i += input_dim) {
        for (int64_t j = 0; j < width; j += input_dim) {
            at::Tensor block = input.index(
                    {
                            idx::Slice(i, std::min(i + static_cast<int64_t>(input_dim), height)),
                            idx::Slice(j, std::min(j + static_cast<int64_t>(input_dim), width))
                    });
            block = block.permute({2, 0, 1}).unsqueeze(0).to(torch::kFloat32).div(255);
            output.push_back(block);
        }
    }
    return output;
}

at::Tensor Model::postprocess(const std::vector<at::Tensor> &input, const cv::Size &output_size) const {
    at::Tensor unblocked = torch::zeros({output_size.height, output_size.width, 3});
    auto input_itr = input.cbegin();
    for (int64_t i = 0; i < output_size.height; i += output_dim) {
        for (int64_t j = 0; j < output_size.width; j += output_dim) {
            unblocked.index_put_(
                    {
                            idx::Slice(i, std::min(i + static_cast<int64_t>(output_dim),
                                                   static_cast<int64_t>(output_size.height))),
                            idx::Slice(j, std::min(j + static_cast<int64_t>(output_dim),
                                                   static_cast<int64_t>(output_size.width)))
                    }, (*input_itr).squeeze(0).permute({1, 2, 0}));
            input_itr++;
        }
    }
    unblocked = torch::clamp((unblocked * 255) + 0.5, 0, 255).to(torch::kUInt8);
    return unblocked;
}

cv::Mat Model::run(cv::Mat input) {
    cv::cvtColor(input, input, cv::COLOR_BGR2RGB);
    int64_t height = input.rows, width = input.cols;
    auto options = torch::TensorOptions().dtype(torch::kUInt8);
    at::Tensor input_t = torch::from_blob(input.data, {height, width, 3}, options);
    std::vector<at::Tensor> blocked_input = preprocess(input_t);
    std::vector<at::Tensor> blocked_output = run(blocked_input);
    at::Tensor output_t = postprocess(blocked_output, cv::Size(width * scale, height * scale));
    cv::Mat output = cv::Mat::ones(output_t.size(0), output_t.size(1), CV_MAKETYPE(cv::DataType<uint8_t>::type, 3));
    auto *output_t_ptr = output_t.data_ptr<uchar>();
    std::memcpy(output.data, output_t_ptr, sizeof(uint8_t) * output_t.numel());
    cv::cvtColor(output, output, cv::COLOR_RGB2BGR);
    return output;
}
