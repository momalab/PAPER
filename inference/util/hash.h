#pragma once

#include <cstdint>
#include <vector>

namespace util
{

struct VectorDoubleHash
{
    std::size_t operator()(const std::vector<double>& v) const;
};

struct VectorDoubleEqual
{
    bool operator()(const std::vector<double>& lhs, const std::vector<double>& rhs) const;
};



} // util