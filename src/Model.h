//
// Created by Armando Herrera on 1/19/21.
//

#ifndef SUPERRESOLUTION_MODEL_H
#define SUPERRESOLUTION_MODEL_H

#include <filesystem>
#include <torch/script.h>
#include <opencv2/core/mat.hpp>

/**
 * Abstracts the libtorch calls and conversions between opencv's Mat and libtorch's Tensors.
 */
class Model {
    //! Stops gradient calculation
    torch::NoGradGuard no_grad;
    //! The torch module that will be used to compute the super resolution.
    torch::jit::script::Module module;
    //! The input dimension size (square)
    size_t input_dim;
    //! The super sampled output dimension size (square).
    size_t output_dim;
    //! The device where to perform the torch operations.
    torch::Device device;

public:
    /**
     * A constructor
     * @param model_path The path to the torchscript model.
     * @param upscale An integer representing the model's scale.
     * @param output_dims_size The output side dimension size.
     */
    Model(const std::filesystem::path &model_path, size_t upscale, size_t output_dims_size);

    /**
     * Performs The Super-Sampling operations
     * @param input An opencv Mat object to super sample.
     * @return The Super-Sampled image.
     */
    //cv::Mat run_block(const cv::Mat &input);

    at::Tensor run(const at::Tensor &input);

    at::Tensor preprocess(const at::Tensor &input) const;

    static at::Tensor postprocess(const at::Tensor &input, const cv::Size &output_size);

    cv::Mat run(const cv::Mat &input);
};


#endif //SUPERRESOLUTION_MODEL_H
