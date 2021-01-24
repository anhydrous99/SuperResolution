//
// Created by constexpr_dog on 1/24/21.
// TODO: Test on windows
//

#ifndef SUPERRESOLUTION_PROGRESSBAR_H
#define SUPERRESOLUTION_PROGRESSBAR_H


class ProgressBar {
    int count, current = 0;

    static int get_term_width();
    void print_progress_bar() const;
public:
    explicit ProgressBar(int count);
    void step();
};


#endif //SUPERRESOLUTION_PROGRESSBAR_H
