#include "random.h"

#include <initializer_list>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
#include "tensor.h"

using namespace std;
using namespace type;

namespace util
{

UniformRealDistribution::UniformRealDistribution(double min_value, double max_value)
{
    generator = std::mt19937(random_device());
    distribution = std::uniform_real_distribution<double>(min_value, std::nextafter(max_value, max_value + 1.0));
}

} // util