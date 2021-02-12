//
// Created by Armando Herrera on 1/20/21.
//

#include "utils.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cctype>
#include <array>

bool check_input_extensions(std::string extension, Glog *glog) {
    std::array<std::string, 10> image_extensions{
            ".bpm", ".dib", ".jpeg", ".jpg", ".jpe",
            "jp2", ".png", ".webp", ".tiff", ".tif"
    };
    std::array<std::string, 2> video_extensions{
            ".avi", ".mp4"
    };
    // Convert all characters in extension to lower case
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    // Check if the extension is that of an image
    bool is_img_ext = std::find(image_extensions.begin(), image_extensions.end(), extension) != image_extensions.end();
    // Check if the extension is that of a video
    bool is_vid_ext = std::find(video_extensions.begin(), video_extensions.end(), extension) != video_extensions.end();
    // Throw error if extension is neither
    glog->Check(is_img_ext || is_vid_ext, "Input file extension is not supported\n");
    return is_img_ext;
}

cv::Mat blend_bicubic(const cv::Mat &input, const cv::Mat &to_blend, size_t scale, size_t weight) {
    cv::Mat cubic_frame, output;
    cv::resize(input, cubic_frame, cv::Size(), static_cast<float>(scale), static_cast<float>(scale), cv::INTER_CUBIC);
    double alpha = static_cast<double>(weight) / 100;
    double beta = (1.0 - alpha);
    cv::addWeighted(cubic_frame, alpha, to_blend, beta, 0.0, output);
    return output;
}
