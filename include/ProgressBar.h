//
// Created by constexpr_dog on 1/24/21.
//

#ifndef SUPERRESOLUTION_PROGRESSBAR_H
#define SUPERRESOLUTION_PROGRESSBAR_H


/**
 * A progress bar class that is used to print a progress bar and shows progress.
 */
class ProgressBar {
    int count, current = 0;
    /**
     * Prints the progress bar
     */
    void print_progress_bar() const;

public:
    /**
     * Compiled differently depending on the platform, gets the width of the terminal.
     */
    static int get_term_width();
    /**
     * The constructor
     * @param count The total number of objects to show progress for
     */
    explicit ProgressBar(int count);
    /**
     * Reports a progressed step and reprint the progress bar
     */
    void step();
};


#endif //SUPERRESOLUTION_PROGRESSBAR_H
