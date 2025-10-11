#pragma once

#include <vector>
#include <utility>

namespace type
{

template <class U>
Tensor<1,double> Tensor<1,double>::copy(const std::vector<U>& data)
{
    Tensor<1,double> r;
    for (const auto& d : data) r._data.emplace_back(d);
    return r;
}

template <class U>
Tensor<1,double> Tensor<1,double>::move(std::vector<U>&& data)
{
    Tensor<1,double> r;
    for (auto& d : data) r._data.emplace_back(std::move(d));
    return r;
}

} // type