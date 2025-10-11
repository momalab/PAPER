#include "tensor1f.h"

#include <algorithm>
#include <cstdlib>
#include <initializer_list>
#include <iterator>
#include <ostream>
#include <utility>
#include <vector>

namespace type
{

float Tensor<1,float>::_precision = 0;

// Tensor<1,float>::Tensor(const std::vector<float>& data) : _data(data) {}

Tensor<1,float>::Tensor(const std::initializer_list<std::size_t>& sizes, const float& value)
    : Tensor(std::vector<std::size_t>(sizes), value) {}

Tensor<1,float>::Tensor(const std::vector<std::size_t>& sizes, const float& value)
    : Tensor(sizes.begin(), sizes.end(), value) {}

Tensor<1,float>::Tensor(std::vector<std::size_t>::const_iterator begin, std::vector<std::size_t>::const_iterator end, const float& value)
{
    if (begin >= end) throw std::runtime_error("Tensor<1,float>::Tensor: invalid size dimension");
    auto size = *begin;
    _data.resize(size, value);
}

float& Tensor<1,float>::operator[](int i)
{
    return _data[i];
}

const float& Tensor<1,float>::operator[](int i) const
{
    return _data[i];
}

Tensor<1,float>& Tensor<1,float>::operator+=(const Tensor<1,float>& a)
{
    for (std::size_t i = 0; i < _data.size(); ++i) _data[i] += a._data[i];
    return *this;
}

Tensor<1,float>& Tensor<1,float>::operator+=(float a)
{
    for (std::size_t i = 0; i < _data.size(); ++i) _data[i] += a;
    return *this;
}

Tensor<1,float>& Tensor<1,float>::operator-=(const Tensor<1,float>& a)
{
    for (std::size_t i = 0; i < _data.size(); ++i) _data[i] -= a._data[i];
    return *this;
}

Tensor<1,float>& Tensor<1,float>::operator-=(float a)
{
    for (std::size_t i = 0; i < _data.size(); ++i) _data[i] -= a;
    return *this;
}

Tensor<1,float>& Tensor<1,float>::operator*=(const Tensor<1,float>& a)
{
    for (std::size_t i = 0; i < _data.size(); ++i) _data[i] *= a._data[i];
    return *this;
}

Tensor<1,float>& Tensor<1,float>::operator*=(float a)
{
    for (std::size_t i = 0; i < _data.size(); ++i) _data[i] *= a;
    return *this;
}

Tensor<1,float> Tensor<1,float>::operator+(const Tensor<1,float>& a) const
{
    Tensor<1,float> r;
    for (std::size_t i = 0; i < _data.size(); ++i) r._data.emplace_back(_data[i] + a._data[i]);
    return r;
}

Tensor<1,float> Tensor<1,float>::operator+(float a) const
{
    Tensor<1,float> r;
    for (std::size_t i = 0; i < _data.size(); ++i) r._data.emplace_back(_data[i] + a);
    return r;
}

Tensor<1,float> Tensor<1,float>::operator-(const Tensor<1,float>& a) const
{
    Tensor<1,float> r;
    for (std::size_t i = 0; i < _data.size(); ++i) r._data.emplace_back(_data[i] - a._data[i]);
    return r;
}

Tensor<1,float> Tensor<1,float>::operator-(float a) const
{
    Tensor<1,float> r;
    for (std::size_t i = 0; i < _data.size(); ++i) r._data.emplace_back(_data[i] - a);
    return r;
}

Tensor<1,float> Tensor<1,float>::operator*(const Tensor<1,float>& a) const
{
    Tensor<1,float> r;
    for (std::size_t i = 0; i < _data.size(); ++i) r._data.emplace_back(_data[i] * a._data[i]);
    return r;
}

Tensor<1,float> Tensor<1,float>::operator*(float a) const
{
    Tensor<1,float> r;
    for (std::size_t i = 0; i < _data.size(); ++i) r._data.emplace_back(_data[i] * a);
    return r;
}

bool Tensor<1,float>::operator==(const Tensor<1,float>& a) const
{
    for (std::size_t i = 0; i < _data.size(); ++i)
        if (abs(_data[i] / a._data[i] - 1.0) > _precision) return false;
    return true;
}

bool Tensor<1,float>::operator==(float a) const
{
    for (std::size_t i = 0; i < _data.size(); ++i)
        if (abs(_data[i] / a - 1.0) > _precision) return false;
    return true;
}

bool Tensor<1,float>::operator!=(const Tensor<1,float>& a) const
{
    return !(*this == a);
}

bool Tensor<1,float>::operator!=(float a) const
{
    return !(*this == a);
}

bool Tensor<1,float>::any_empty() const
{
    return empty();
}

size_t Tensor<1,float>::argmax() const
{
    if (empty()) throw std::runtime_error("Tensor<1,float>::argmax: empty tensor");
    return std::distance(_data.begin(), std::max_element(_data.begin(), _data.end()));
}

float& Tensor<1,float>::back()
{
    return _data.back();
}

const float& Tensor<1,float>::back() const
{
    return _data.back();
}

typename std::vector<float>::iterator Tensor<1,float>::begin()
{
    return _data.begin();
}

typename std::vector<float>::const_iterator Tensor<1,float>::begin() const
{
    return _data.begin();
}

Tensor<1,float> Tensor<1,float>::copy(const std::vector<float>& data)
{
    Tensor<1,float> r;
    r._data = data;
    return r;
}

void Tensor<1,float>::emplace_back(const float& value)
{
    _data.emplace_back(value);
}

void Tensor<1,float>::emplace_back(float&& value)
{
    _data.emplace_back(std::move(value));
}

bool Tensor<1,float>::empty() const
{
    return _data.empty();
}

typename std::vector<float>::iterator Tensor<1,float>::end()
{
    return _data.end();
}

typename std::vector<float>::const_iterator Tensor<1,float>::end() const
{
    return _data.end();
}

float& Tensor<1,float>::front()
{
    return _data.front();
}

const float& Tensor<1,float>::front() const
{
    return _data.front();
}

bool Tensor<1,float>::front_empty() const
{
    return empty();
}

Tensor<1,float> Tensor<1,float>::move(std::vector<float>&& data)
{
    Tensor<1,float> r;
    r._data = std::move(data);
    return r;
}

void Tensor<1,float>::push_back(const float& value)
{
    _data.push_back(value);
}

void Tensor<1,float>::push_back(float&& value)
{
    _data.push_back(std::move(value));
}

void Tensor<1,float>::resize(std::size_t n)
{
    _data.resize(n);
}

void Tensor<1,float>::resize(std::size_t n, const float& value)
{
    _data.resize(n, value);
}

std::size_t Tensor<1,float>::size() const
{
    return _data.size();
}

std::vector<std::size_t> Tensor<1,float>::shape() const
{
    return {_data.size()};
}

std::vector<float>& Tensor<1,float>::vector()
{
    return _data;
}

const std::vector<float>& Tensor<1,float>::vector() const
{
    return _data;
}

float Tensor<1,float>::precision()
{
    return _precision;
}

float Tensor<1,float>::precision(float p)
{
    return _precision = p;
}

} // type