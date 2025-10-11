#pragma once

#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fhe
{

const uint64_t DEFAULT_SCALE_BITSIZE = 20ULL;
const double DEFAULT_SCALE = 1ULL << DEFAULT_SCALE_BITSIZE;
const uint64_t SUBLEVELS = 2ULL;
const uint64_t DEFAULT_BITSIZE = DEFAULT_SCALE_BITSIZE * SUBLEVELS;

const std::unordered_map<std::size_t, std::pair<std::vector<int>, double>> DEFAULT_PARAMS = []
{
    std::unordered_map<std::size_t, std::pair<std::vector<int>, double>> map;

    auto make_vector = [](std::size_t levels)
    {
        std::vector<int> v(levels, DEFAULT_BITSIZE);
        if (!v.empty()) v.back() = DEFAULT_SCALE_BITSIZE;
        return v;
    };

    map[1ULL << 10] = { make_vector( 1), DEFAULT_SCALE };
    map[1ULL << 11] = { make_vector( 1), DEFAULT_SCALE };
    map[1ULL << 12] = { make_vector( 2), DEFAULT_SCALE };
    map[1ULL << 13] = { make_vector( 4), DEFAULT_SCALE };
    map[1ULL << 14] = { make_vector(10), DEFAULT_SCALE };
    map[1ULL << 15] = { make_vector(22), DEFAULT_SCALE };
    map[1ULL << 16] = { make_vector(34), DEFAULT_SCALE };

    return map;
}();

} // fhe