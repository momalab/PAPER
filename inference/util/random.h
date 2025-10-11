#pragma once

#include <initializer_list>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>
#include "tensor.h"

namespace util
{

struct UniformRealDistribution
{
    std::random_device random_device;
    std::mt19937 generator;
    std::uniform_real_distribution<double> distribution;

    UniformRealDistribution(double min_value, double max_value);
};

template <int D> inline type::Tensor<D,double>
random_tensor(const std::initializer_list<std::size_t>& dimensions, double min_value, double max_value, std::shared_ptr<UniformRealDistribution> dis_ptr = nullptr)
{
    return random_tensor<D>(std::vector<std::size_t>(dimensions), min_value, max_value, dis_ptr);
}

template <int D> type::Tensor<D,double>
random_tensor(const std::vector<size_t>& dimensions, double min_value, double max_value, std::shared_ptr<UniformRealDistribution> dis_ptr = nullptr)
{
    if (dimensions.empty() || dimensions.size() != D) throw std::invalid_argument("util::random_tensor: invalid primary dimensions");
    for (auto& e : dimensions) if (e == 0) throw std::invalid_argument("util::random_tensor: invalid secondary dimensions");
    if (min_value >= max_value) throw std::invalid_argument("util::random_tensor: min_value greater than max_value");

    if (!dis_ptr) dis_ptr = std::make_shared<UniformRealDistribution>(min_value, max_value);
    auto& gen = dis_ptr->generator;
    auto& dis = dis_ptr->distribution;

    type::Tensor<D,double> tensor{dimensions};
    std::vector<size_t> sub_dimensions(dimensions.begin() + 1, dimensions.end());
    if constexpr (D > 1) for (auto& e : tensor) e = random_tensor<D-1>(sub_dimensions, min_value, max_value, dis_ptr);
    else for (auto& e : tensor) e = dis(gen);

    return tensor;
}

} // util