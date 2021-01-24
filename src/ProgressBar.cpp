//
// Created by constexpr_dog on 1/24/21.
//

#include "ProgressBar.h"
#include <iostream>
#include <iomanip>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/ioctl.h>
#include <unistd.h>

int ProgressBar::get_term_width() {
    struct winsize w{};
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    return w.ws_col - 7;
}
#elif defined(_WIN32) || defined(_WIN64)
#include <windows.h>

int ProgressBar::get_term_width() {
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
    return (csbi.srWindow.Right - csbi.srWindow.Left + 1) - 7;
}
#endif

void ProgressBar::print_progress_bar() const {
    int barWidth = get_term_width();
    float progress = static_cast<float>(current) / static_cast<float>(count);
    std::cout << "[";
    int pos = static_cast<int>(static_cast<float>(barWidth) * progress);
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << std::setw(3) << static_cast<int>(progress * 100.f) << "%\r";
    std::cout.flush();
}

ProgressBar::ProgressBar(int count) : count(count) {
    print_progress_bar();
}

void ProgressBar::step() {
    current++;
    print_progress_bar();
}
