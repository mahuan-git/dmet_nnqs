#pragma once

#include <iostream>
#include <chrono>

class Timer {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    bool is_running = false;

public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
        is_running = true;
    }

    void stop(std::string mess="") {
#ifndef DEBUG_LOCAL_ENERGY
        return;
#endif
        if (!is_running) {
            std::cerr << "Timer was not started." << std::endl;
            return;
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto start = std::chrono::time_point_cast<std::chrono::microseconds>(start_time).time_since_epoch().count();
        auto end = std::chrono::time_point_cast<std::chrono::microseconds>(end_time).time_since_epoch().count();

        auto duration = end - start;
        double ms = duration * 0.001;

        std::cout << "[" << mess << "] duration: " << ms << " ms" << std::endl;
        is_running = false;
    }
};

#ifdef TEST
int main() {
    Timer timer[2];
    timer[0].start();

    for (int i = 0; i < 1000000; i++) {
    }

    timer[0].stop("part1");

    return 0;
}
#endif
