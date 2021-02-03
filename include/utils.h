//
// Created by Armando Herrera on 1/20/21.
//

#ifndef SUPERRESOLUTION_UTILS_H
#define SUPERRESOLUTION_UTILS_H

#include <iostream>
#include <iterator>
#include "Glog.h"

/**
 * Checks whether an extension is supported and whether it is a video or image extension.
 * @param extension
 * @return A boolean where true is an image and false is a video
 */
bool check_input_extensions(std::string extension, Glog* glog);

#endif //SUPERRESOLUTION_UTILS_H
