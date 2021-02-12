//
// Created by Armando Herrera on 1/19/21.
//

#ifndef SUPERRESOLUTION_MODEL_H
#define SUPERRESOLUTION_MODEL_H

#include "Glog.h"
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
    int64_t input_dim;
    //! The super sampled output dimension size (square).
    int64_t output_dim;
    //! The output scale compared to the input
    int64_t scale;
    //! The batch size (number of blocks to processes at the same time)
    int64_t batch_size;
    //! The device to perform the torch operations.
    torch::Device device;
    Glog *glog;

public:
    /**
     * A constructor
     * @param model_path The path to the torchscript model.
     * @param upscale An integer representing the model's scale.
     * @param output_size The output side dimension size.
     * @param batch_size The batch size (number of blocks to processes at the same time)
     * @param glog A pointer to the glog object, used for error reporting and tracing
     */
    Model(const std::filesystem::path &model_path, int64_t upscale, int64_t output_size, int64_t batch_size, Glog *glog);

    /**
     * Runs the model against blocks of size output_size/scale and outputs blocks of size output_size
     * @param input The input blocks
     * @return The output blocks
     */
    std::vector<at::Tensor> run(const std::vector<at::Tensor> &input);

    /**
     * Performs preprocessing on an image by converting it from uint8 to float, scaled to what the model was trained on
     * and split into blocks.
     * @param input The input image
     * @return The blocked output tensors
     */
    std::vector<at::Tensor> preprocess(const at::Tensor &input) const;

    /**
     * Converts the output blocks, from the model, into the output, super-sampled, image.
     * @param input The inferenced blocks
     * @param output_size The shape of the output image
     * @return The image
     */
    at::Tensor postprocess(const std::vector<at::Tensor> &input, const cv::Size &output_size) const;

    /**
     * The main inferencing function, super-samples an opencv image
     * @param input The input image
     * @return The output super-sampled image
     */
    cv::Mat run(cv::Mat input);
};


#endif //SUPERRESOLUTION_MODEL_H
