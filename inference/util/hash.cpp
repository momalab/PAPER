#include "hash.h"

#include <cstdint>
#include <vector>

using namespace std;

namespace util
{

size_t VectorDoubleHash::operator()(const vector<double>& v) const
{
    size_t seed = v.size(); // include size in hash to avoid collisions
    for (size_t i = 0; i < v.size(); ++i)
    {
        size_t h = hash<double>{}(v[i]);
        seed ^= h + 0x9e3779b9 + (seed << 6) + (seed >> 2); // boost-style hash combine
    }
    return seed;
}

bool VectorDoubleEqual::operator()(const vector<double>& lhs, const vector<double>& rhs) const
{
    return lhs == rhs; // vector provides exact comparison
}

} // util