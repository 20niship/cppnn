#pragma once
#include <array>
#include <cassert>
#include <iostream>
#include <chrono>

/* #define MU_ASSERT(A) assert((A) "MovUtl assertion failed!  : " #A  __FILE__ __LINE__ ) */
#define MU_ASSERT(A) assert(A)

#define DISP(A) std::cout << #A << " = " << (A) << std::endl

// clang-format off
#define DURATION(A) \
{ \
    const auto start = std::chrono::system_clock::now(); \
    {A}; \
    const auto end      = std::chrono::system_clock::now(); \
    const auto cnt = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); \
    std::cout << "duration = " <<#A << "  : " << cnt << std::endl; \
}
// clang-format on
