//
// Created by Armando Herrera on 1/19/21.
//

#include "Model.h"
#include "utils.h"

#include <glog/logging.h>
#include <algorithm>
#include <deque>


namespace idx = torch::indexing;

Model::Model(const std::filesystem::path &model_path, size_t upscale, size_t output_dims_size) : device("cpu") {
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

    input_dim = output_dims_size / upscale;
    output_dim = output_dims_size;
    scale = upscale;
}

//at::Tensor Model::run(const at::Tensor &input) {
//    at::Tensor output_tensor = torch::zeros({input.size(0), 3, static_cast<long>(output_dim), static_cast<long>(output_dim)}, torch::Device("cpu"));
//    for (int64_t i = 0; i < input.size(0); i++) {
//        std::cout << i << std::endl;
//        auto partial_input = input.index({i, "..."}).unsqueeze(0).to(device);
//        auto partial_output = module.forward({partial_input}).toTensor().to(torch::Device("cpu")).squeeze(0);
//        output_tensor.index_put_({i, "..."}, partial_output);
//    }
//    return input;
//}

std::vector<at::Tensor> Model::run(const std::vector<at::Tensor> &input) {
    std::vector<at::Tensor> output_tensors;
    for (const at::Tensor &tensor : input)
        output_tensors.push_back(module.forward({tensor.to(torch::Device(device))}).toTensor().to(torch::Device("cpu")));
    return output_tensors;
}

std::vector<at::Tensor> Model::preprocess(const at::Tensor &input) const {
    int64_t width = input.size(0), height = input.size(1);
    std::vector<at::Tensor> output;
    for (int64_t i = 0; i < width; i += input_dim) {
        for (int64_t j = 0; j < height; j += input_dim) {
            at::Tensor block = input.index(
                    {
                            idx::Slice(i, std::min(i + static_cast<int64_t>(input_dim), width)),
                            idx::Slice(j, std::min(j + static_cast<int64_t>(input_dim), width))
                    });
            output.push_back(block.permute({2, 0, 1}).unsqueeze(0).to(torch::kFloat32).div(255));
        }
    }
    return output;
}

at::Tensor Model::postprocess(const std::vector<at::Tensor> &input, const cv::Size &output_size) {
    at::Tensor unblocked = torch::zeros({output_size.width, output_size.height, 3});
    auto input_itr = input.cbegin();
    for (int64_t i = 0; i < output_size.width; i += output_dim) {
        for (int64_t j = 0; j < output_size.height; j += output_dim) {
            unblocked.index_put_(
                    {
                        idx::Slice(i, std::min(i + static_cast<int64_t>(input_dim), static_cast<int64_t>(output_size.width))),
                        idx::Slice(j, std::min(j + static_cast<int64_t>(input_dim), static_cast<int64_t>(output_size.height)))
                        }, (*input_itr).unsqueeze(0).permute({1, 2, 0}));
            input_itr++;
        }
    }
    //unblocked = torch::clamp((output * 255) + 0.5, 0, 255).permute({1, 2, 0}).to
    unblocked = torch::clamp((unblocked * 255) + 0.5, 0, 255).to(torch::kUInt8);
    return unblocked;
}

cv::Mat Model::run(const cv::Mat &input) {
    int64_t width = input.rows, height = input.cols;
    at::Tensor input_t = torch::from_blob(input.data, {width, height, 3});
    std::vector<at::Tensor> blocked_input = preprocess(input_t);
    std::vector<at::Tensor> blocked_output = run(blocked_input);
    at::Tensor output_t = postprocess(blocked_output, cv::Size(width * scale, height * scale));
//    output_t = postprocess(output_t, cv::Size(width * 4, height * 4));
//    auto *output_ptr = output_t.data_ptr<uint8_t>();
//    return cv::Mat(cv::Size(output_t.size(2), output_t.size(3)), CV_8UC3, output_ptr);
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
