//
// Created by constexpr_dog on 1/24/21.
//

#ifndef SUPERRESOLUTION_PROGRESSBAR_H
#define SUPERRESOLUTION_PROGRESSBAR_H


class ProgressBar {
    int count, current = 0;
    void print_progress_bar() const;

public:
    static int get_term_width();
    explicit ProgressBar(int count);
    void step();
};


#endif //SUPERRESOLUTION_PROGRESSBAR_H
