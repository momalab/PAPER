#pragma once

#include <cstddef>
#include <thread>

namespace config
{

#ifdef THREADS
constexpr std::size_t NTHREADS = THREADS;
#else
const std::size_t NTHREADS = std::thread::hardware_concurrency();
#endif

} // config