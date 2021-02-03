//
// Created by Armando Herrera on 1/20/21.
//

#include "utils.h"
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
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    bool is_img_ext = std::find(image_extensions.begin(), image_extensions.end(), extension) != image_extensions.end();
    bool is_vid_ext = std::find(video_extensions.begin(), video_extensions.end(), extension) != video_extensions.end();
    glog->Check(is_img_ext || is_vid_ext, "Input file extension is not supported\n");
    return is_img_ext;
}
