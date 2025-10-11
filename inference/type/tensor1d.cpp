#include "tensor1d.h"

#include <algorithm>
#include <cstdlib>
#include <initializer_list>
#include <iterator>
#include <ostream>
#include <utility>
#include <vector>

namespace type
{

double Tensor<1,double>::_precision = 0;

// Tensor<1,double>::Tensor(const std::vector<double>& data) : _data(data) {}

Tensor<1,double>::Tensor(const std::initializer_list<std::size_t>& sizes, const double& value)
    : Tensor(std::vector<std::size_t>(sizes), value) {}

Tensor<1,double>::Tensor(const std::vector<std::size_t>& sizes, const double& value)
    : Tensor(sizes.begin(), sizes.end(), value) {}

Tensor<1,double>::Tensor(std::vector<std::size_t>::const_iterator begin, std::vector<std::size_t>::const_iterator end, const double& value)
{
    if (begin >= end) throw std::runtime_error("Tensor<1,double>::Tensor: invalid size dimension");
    auto size = *begin;
    _data.resize(size, value);
}

double& Tensor<1,double>::operator[](int i)
{
    return _data[i];
}

const double& Tensor<1,double>::operator[](int i) const
{
    return _data[i];
}

Tensor<1,double>& Tensor<1,double>::operator+=(const Tensor<1,double>& a)
{
    for (std::size_t i = 0; i < _data.size(); ++i) _data[i] += a._data[i];
    return *this;
}

Tensor<1,double>& Tensor<1,double>::operator+=(double a)
{
    for (std::size_t i = 0; i < _data.size(); ++i) _data[i] += a;
    return *this;
}

Tensor<1,double>& Tensor<1,double>::operator-=(const Tensor<1,double>& a)
{
    for (std::size_t i = 0; i < _data.size(); ++i) _data[i] -= a._data[i];
    return *this;
}

Tensor<1,double>& Tensor<1,double>::operator-=(double a)
{
    for (std::size_t i = 0; i < _data.size(); ++i) _data[i] -= a;
    return *this;
}

Tensor<1,double>& Tensor<1,double>::operator*=(const Tensor<1,double>& a)
{
    for (std::size_t i = 0; i < _data.size(); ++i) _data[i] *= a._data[i];
    return *this;
}

Tensor<1,double>& Tensor<1,double>::operator*=(double a)
{
    for (std::size_t i = 0; i < _data.size(); ++i) _data[i] *= a;
    return *this;
}

Tensor<1,double> Tensor<1,double>::operator+(const Tensor<1,double>& a) const
{
    Tensor<1,double> r;
    for (std::size_t i = 0; i < _data.size(); ++i) r._data.emplace_back(_data[i] + a._data[i]);
    return r;
}

Tensor<1,double> Tensor<1,double>::operator+(double a) const
{
    Tensor<1,double> r;
    for (std::size_t i = 0; i < _data.size(); ++i) r._data.emplace_back(_data[i] + a);
    return r;
}

Tensor<1,double> Tensor<1,double>::operator-(const Tensor<1,double>& a) const
{
    Tensor<1,double> r;
    for (std::size_t i = 0; i < _data.size(); ++i) r._data.emplace_back(_data[i] - a._data[i]);
    return r;
}

Tensor<1,double> Tensor<1,double>::operator-(double a) const
{
    Tensor<1,double> r;
    for (std::size_t i = 0; i < _data.size(); ++i) r._data.emplace_back(_data[i] - a);
    return r;
}

Tensor<1,double> Tensor<1,double>::operator*(const Tensor<1,double>& a) const
{
    Tensor<1,double> r;
    for (std::size_t i = 0; i < _data.size(); ++i) r._data.emplace_back(_data[i] * a._data[i]);
    return r;
}

Tensor<1,double> Tensor<1,double>::operator*(double a) const
{
    Tensor<1,double> r;
    for (std::size_t i = 0; i < _data.size(); ++i) r._data.emplace_back(_data[i] * a);
    return r;
}

bool Tensor<1,double>::operator==(const Tensor<1,double>& a) const
{
    for (std::size_t i = 0; i < _data.size(); ++i)
        if (abs(_data[i] / a._data[i] - 1.0) > _precision) return false;
    return true;
}

bool Tensor<1,double>::operator==(double a) const
{
    for (std::size_t i = 0; i < _data.size(); ++i)
        if (abs(_data[i] / a - 1.0) > _precision) return false;
    return true;
}

bool Tensor<1,double>::operator!=(const Tensor<1,double>& a) const
{
    return !(*this == a);
}

bool Tensor<1,double>::operator!=(double a) const
{
    return !(*this == a);
}

bool Tensor<1,double>::any_empty() const
{
    return empty();
}

size_t Tensor<1,double>::argmax() const
{
    if (empty()) throw std::runtime_error("Tensor<1,double>::argmax: empty tensor");
    return std::distance(_data.begin(), std::max_element(_data.begin(), _data.end()));
}

double& Tensor<1,double>::back()
{
    return _data.back();
}

const double& Tensor<1,double>::back() const
{
    return _data.back();
}

typename std::vector<double>::iterator Tensor<1,double>::begin()
{
    return _data.begin();
}

typename std::vector<double>::const_iterator Tensor<1,double>::begin() const
{
    return _data.begin();
}

Tensor<1,double> Tensor<1,double>::copy(const std::vector<double>& data)
{
    Tensor<1,double> r;
    r._data = data;
    return r;
}

void Tensor<1,double>::emplace_back(const double& value)
{
    _data.emplace_back(value);
}

void Tensor<1,double>::emplace_back(double&& value)
{
    _data.emplace_back(std::move(value));
}

bool Tensor<1,double>::empty() const
{
    return _data.empty();
}

typename std::vector<double>::iterator Tensor<1,double>::end()
{
    return _data.end();
}

typename std::vector<double>::const_iterator Tensor<1,double>::end() const
{
    return _data.end();
}

double& Tensor<1,double>::front()
{
    return _data.front();
}

const double& Tensor<1,double>::front() const
{
    return _data.front();
}

bool Tensor<1,double>::front_empty() const
{
    return empty();
}

Tensor<1,double> Tensor<1,double>::move(std::vector<double>&& data)
{
    Tensor<1,double> r;
    r._data = std::move(data);
    return r;
}

void Tensor<1,double>::push_back(const double& value)
{
    _data.push_back(value);
}

void Tensor<1,double>::push_back(double&& value)
{
    _data.push_back(std::move(value));
}

void Tensor<1,double>::resize(std::size_t n)
{
    _data.resize(n);
}

void Tensor<1,double>::resize(std::size_t n, const double& value)
{
    _data.resize(n, value);
}

std::size_t Tensor<1,double>::size() const
{
    return _data.size();
}

std::vector<std::size_t> Tensor<1,double>::shape() const
{
    return {_data.size()};
}

std::vector<double>& Tensor<1,double>::vector()
{
    return _data;
}

const std::vector<double>& Tensor<1,double>::vector() const
{
    return _data;
}

double Tensor<1,double>::precision()
{
    return _precision;
}

double Tensor<1,double>::precision(double p)
{
    return _precision = p;
}

} // type