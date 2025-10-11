#pragma once

#include <vector>
#include <utility>

namespace type
{

template <class U>
Tensor<1,float> Tensor<1,float>::copy(const std::vector<U>& data)
{
    Tensor<1,float> r;
    for (const auto& d : data) r._data.emplace_back(d);
    return r;
}

template <class U>
Tensor<1,float> Tensor<1,float>::move(std::vector<U>&& data)
{
    Tensor<1,float> r;
    for (auto& d : data) r._data.emplace_back(std::move(d));
    return r;
}

} // type