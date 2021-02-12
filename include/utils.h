//
// Created by Armando Herrera on 1/20/21.
//

#ifndef SUPERRESOLUTION_UTILS_H
#define SUPERRESOLUTION_UTILS_H

#include <iostream>
#include <iterator>
#include <opencv2/core/mat.hpp>
#include "Glog.h"

/**
 * Checks whether an extension is supported and whether it is a video or image extension.
 * @param extension
 * @param A glog object pointer used for error reporting and tracing
 * @return A boolean where true is an image and false is a video
 */
bool check_input_extensions(std::string extension, Glog* glog);

/**
 * Blends an input frame with the interpolated frame of another
 * @param input The input frame
 * @param to_blend The frame to blend whose side shape is (initial size)*scale
 * @param scale The amount to scale the original image
 * @param weight The weight to blend the frame
 * @return The blended frame
 */
cv::Mat blend_bicubic(const cv::Mat &input, const cv::Mat &to_blend, size_t scale, size_t weight);

#endif //SUPERRESOLUTION_UTILS_H
