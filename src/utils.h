//
// Created by Armando Herrera on 1/20/21.
//

#ifndef SUPERRESOLUTION_UTILS_H
#define SUPERRESOLUTION_UTILS_H

#include <iostream>
#include <iterator>

/**
 * Checks whether an extension is supported and whether it is a video or image extension.
 * @param extension
 * @return A boolean where true is an image and false is a video
 */
bool check_input_extensions(std::string extension);

template<typename Iter>
void print_arr(Iter begin, Iter end) {
    typedef typename std::iterator_traits<Iter>::difference_type diff_type;
    for (; begin < end; begin++) {
        std::cout << *begin << ' ';
    }
    std::cout << std::endl;
}

#endif //SUPERRESOLUTION_UTILS_H
