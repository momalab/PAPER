#pragma once

#include "tensordt.h"

#include <initializer_list>
#include <ostream>
#include <type_traits>
#include <utility>
#include <vector>

namespace type
{

// Tensor<D,T>

// template <int D, class T>
// Tensor<D,T>::Tensor(const std::vector<Tensor<D-1,T>>& data) : _data(data) {}

template <int D, class T>
Tensor<D,T>::Tensor(const std::initializer_list<std::size_t>& sizes, const T& value)
    : Tensor(std::vector<std::size_t>(sizes), value) {}

template <int D, class T>
Tensor<D,T>::Tensor(const std::vector<std::size_t>& sizes, const T& value)
    : Tensor(sizes.begin(), sizes.end(), value) {}

template <int D, class T>
Tensor<D,T>::Tensor(std::vector<std::size_t>::const_iterator begin, std::vector<std::size_t>::const_iterator end, const T& value)
{
    auto size = *begin;
    _data.resize(size);
    for (auto& tensor : _data) tensor = Tensor<D-1,T>(begin + 1, end, value);
}

template <int D, class T>
Tensor<D-1,T>& Tensor<D,T>::operator[](int i)
{
    return _data[i];
}

template <int D, class T>
const Tensor<D-1,T>& Tensor<D,T>::operator[](int i) const
{
    return _data[i];
}

template <int D, class T>
template <class U>
Tensor<D,T>& Tensor<D,T>::operator+=(const Tensor<D,U>& a)
{
    for (std::size_t i = 0; i < _data.size(); ++i) _data[i] += a.vector()[i];
    return *this;
}

template <int D, class T>
template <class U, typename std::enable_if<!std::is_same<Tensor<D,U>, typename std::decay<U>::type>::value, int>::type>
Tensor<D,T>& Tensor<D,T>::operator+=(const U& a)
{
    for (std::size_t i = 0; i < _data.size(); ++i) _data[i] += a;
    return *this;
}

template <int D, class T>
template <class U>
Tensor<D,T>& Tensor<D,T>::operator-=(const Tensor<D,U>& a)
{
    for (std::size_t i = 0; i < _data.size(); ++i) _data[i] -= a.vector()[i];
    return *this;
}

template <int D, class T>
template <class U, typename std::enable_if<!std::is_same<Tensor<D,U>, typename std::decay<U>::type>::value, int>::type>
Tensor<D,T>& Tensor<D,T>::operator-=(const U& a)
{
    for (std::size_t i = 0; i < _data.size(); ++i) _data[i] -= a;
    return *this;
}

template <int D, class T>
template <class U>
Tensor<D,T>& Tensor<D,T>::operator*=(const Tensor<D,U>& a)
{
    for (std::size_t i = 0; i < _data.size(); ++i) _data[i] *= a.vector()[i];
    return *this;
}

template <int D, class T>
template <class U, typename std::enable_if<!std::is_same<Tensor<D,U>, typename std::decay<U>::type>::value, int>::type>
Tensor<D,T>& Tensor<D,T>::operator*=(const U& a)
{
    for (std::size_t i = 0; i < _data.size(); ++i) _data[i] *= a;
    return *this;
}

template <int D, class T>
template <class U>
Tensor<D,T> Tensor<D,T>::operator+(const Tensor<D,U>& a) const
{
    Tensor<D,T> r;
    for (std::size_t i = 0; i < _data.size(); ++i) r._data.emplace_back(_data[i] + a.vector()[i]);
    return r;
}

template <int D, class T>
template <class U, typename std::enable_if<!std::is_same<Tensor<D,U>, typename std::decay<U>::type>::value, int>::type>
Tensor<D,T> Tensor<D,T>::operator+(const U& a) const
{
    Tensor<D,T> r;
    for (std::size_t i = 0; i < _data.size(); ++i) r._data.emplace_back(_data[i] + a);
    return r;
}

template <int D, class T>
template <class U>
Tensor<D,T> Tensor<D,T>::operator-(const Tensor<D,U>& a) const
{
    Tensor<D,T> r;
    for (std::size_t i = 0; i < _data.size(); ++i) r._data.emplace_back(_data[i] - a.vector()[i]);
    return r;
}

template <int D, class T>
template <class U, typename std::enable_if<!std::is_same<Tensor<D,U>, typename std::decay<U>::type>::value, int>::type>
Tensor<D,T> Tensor<D,T>::operator-(const U& a) const
{
    Tensor<D,T> r;
    for (std::size_t i = 0; i < _data.size(); ++i) r._data.emplace_back(_data[i] - a);
    return r;
}

template <int D, class T>
template <class U>
Tensor<D,T> Tensor<D,T>::operator*(const Tensor<D,U>& a) const
{
    Tensor<D,T> r;
    for (std::size_t i = 0; i < _data.size(); ++i) r._data.emplace_back(_data[i] * a.vector()[i]);
    return r;
}

template <int D, class T>
template <class U, typename std::enable_if<!std::is_same<Tensor<D,U>, typename std::decay<U>::type>::value, int>::type>
Tensor<D,T> Tensor<D,T>::operator*(const U& a) const
{
    Tensor<D,T> r;
    for (std::size_t i = 0; i < _data.size(); ++i) r._data.emplace_back(_data[i] * a);
    return r;
}

template <int D, class T>
bool Tensor<D,T>::operator==(const Tensor<D,T>& a) const
{
    return _data == a._data;
}

template <int D, class T>
bool Tensor<D,T>::operator==(const T& a) const
{
    for (const auto& tensor : _data)
        if (tensor != a) return false;
    return true;    
}

template <int D, class T>
bool Tensor<D,T>::operator!=(const Tensor<D,T>& a) const
{
    return _data != a._data;
}

template <int D, class T>
bool Tensor<D,T>::operator!=(const T& a) const
{
    return !(*this == a);
}

template <int D, class T>
bool Tensor<D,T>::any_empty() const
{
    if (empty()) return true;
    for (const auto& tensor : _data)
        if (tensor.any_empty()) return true;
    return false;
}

template <int D, class T>
Tensor<D-1,T>& Tensor<D,T>::back()
{
    return _data.back();
}

template <int D, class T>
const Tensor<D-1,T>& Tensor<D,T>::back() const
{
    return _data.back();
}

template <int D, class T>
typename std::vector<Tensor<D-1,T>>::iterator Tensor<D,T>::begin()
{
    return _data.begin();
}

template <int D, class T>
typename std::vector<Tensor<D-1,T>>::const_iterator Tensor<D,T>::begin() const
{
    return _data.begin();
}

template <int D, class T>
Tensor<D,T> Tensor<D,T>::copy(const std::vector<Tensor<D-1,T>>& data)
{
    Tensor<D,T> r;
    r._data = data;
    return r;
}

template <int D, class T>
template <class U>
Tensor<D,T> Tensor<D,T>::copy(const std::vector<U>& data)
{
    Tensor<D,T> r;
    for (const auto& u : data) r._data.emplace_back(Tensor<D-1,T>::copy(u));
    return r;
}

template <int D, class T>
void Tensor<D,T>::emplace_back(const Tensor<D-1,T>& tensor)
{
    _data.emplace_back(tensor);
}

template <int D, class T>
void Tensor<D,T>::emplace_back(Tensor<D-1,T>&& tensor)
{
    _data.emplace_back(std::move(tensor));
}

template <int D, class T>
bool Tensor<D,T>::empty() const
{
    return _data.empty();
}

template <int D, class T>
typename std::vector<Tensor<D-1,T>>::iterator Tensor<D,T>::end()
{
    return _data.end();
}

template <int D, class T>
typename std::vector<Tensor<D-1,T>>::const_iterator Tensor<D,T>::end() const
{
    return _data.end();
}

template <int D, class T>
Tensor<D-1,T>& Tensor<D,T>::front()
{
    return _data.front();
}

template <int D, class T>
const Tensor<D-1,T>& Tensor<D,T>::front() const
{
    return _data.front();
}

template <int D, class T>
bool Tensor<D,T>::front_empty() const
{
    if (empty()) return true;
    return _data.front().front_empty();
}

template <int D, class T>
Tensor<D,T> Tensor<D,T>::move(std::vector<Tensor<D-1,T>>&& data)
{
    Tensor<D,T> r;
    r._data = std::move(data);
    return r;
}

template <int D, class T>
template <class U>
Tensor<D,T> Tensor<D,T>::move(std::vector<U>&& data)
{
    Tensor<D,T> r;
    for (auto& u : data) r._data.emplace_back(Tensor<D-1,T>::move(std::move(u)));
    return r;
}

template <int D, class T>
void Tensor<D,T>::push_back(const Tensor<D-1,T>& tensor)
{
    _data.push_back(tensor);
}

template <int D, class T>
void Tensor<D,T>::push_back(Tensor<D-1,T>&& tensor)
{
    _data.push_back(std::move(tensor));
}

template <int D, class T>
void Tensor<D,T>::resize(std::size_t n)
{
    _data.resize(n);
}

template <int D, class T>
void Tensor<D,T>::resize(std::size_t n, const Tensor<D-1,T>& tensor)
{
    _data.resize(n, tensor);
}

template <int D, class T>
std::size_t Tensor<D,T>::size() const
{
    return _data.size();
}

template <int D, class T>
std::vector<std::size_t> Tensor<D,T>::shape() const
{
    std::vector<std::size_t> sizes;
    sizes.push_back(_data.size());
    if (!_data.empty())
    {
        auto sub_sizes = _data.front().shape();
        sizes.insert(sizes.end(), sub_sizes.begin(), sub_sizes.end());
    }
    return sizes;
}

template <int D, class T>
std::vector<Tensor<D-1,T>>& Tensor<D,T>::vector()
{
    return _data;
}

template <int D, class T>
const std::vector<Tensor<D-1,T>>& Tensor<D,T>::vector() const
{
    return _data;
}

} // type