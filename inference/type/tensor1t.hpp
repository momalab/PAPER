#pragma once

#include "tensor1t.h"

#include <initializer_list>
#include <ostream>
#include <type_traits>
#include <utility>
#include <vector>

namespace type
{

// template <class T>
// Tensor<1,T>::Tensor(const std::vector<T>& data) : _data(data) {}

template <class T>
Tensor<1,T>::Tensor(const std::initializer_list<std::size_t>& sizes, const T& value)
    : Tensor(std::vector<std::size_t>(sizes), value) {}

template <class T>
Tensor<1,T>::Tensor(const std::vector<std::size_t>& sizes, const T& value)
    : Tensor(sizes.begin(), sizes.end(), value) {}

template <class T>
Tensor<1,T>::Tensor(std::vector<std::size_t>::const_iterator begin, std::vector<std::size_t>::const_iterator end, const T& value)
{
    if (begin >= end) throw std::runtime_error("Tensor<1,T>::Tensor: invalid size dimension");
    auto size = *begin;
    _data.resize(size, value);
}

template <class T>
T& Tensor<1,T>::operator[](int i)
{
    return _data[i];
}

template <class T>
const T& Tensor<1,T>::operator[](int i) const
{
    return _data[i];
}

template <class T>
template <class U>
Tensor<1,T>& Tensor<1,T>::operator+=(const Tensor<1,U>& a)
{
    for (std::size_t i = 0; i < _data.size(); ++i) _data[i] += a.vector()[i];
    return *this;
}

template <class T>
template <class U, typename std::enable_if<!std::is_same<Tensor<1,U>, typename std::decay<U>::type>::value, int>::type>
Tensor<1,T>& Tensor<1,T>::operator+=(const U& a)
{
    for (std::size_t i = 0; i < _data.size(); ++i) _data[i] += a;
    return *this;
}

template <class T>
template <class U>
Tensor<1,T>& Tensor<1,T>::operator-=(const Tensor<1,U>& a)
{
    for (std::size_t i = 0; i < _data.size(); ++i) _data[i] -= a.vector()[i];
    return *this;
}

template <class T>
template <class U, typename std::enable_if<!std::is_same<Tensor<1,U>, typename std::decay<U>::type>::value, int>::type>
Tensor<1,T>& Tensor<1,T>::operator-=(const U& a)
{
    for (std::size_t i = 0; i < _data.size(); ++i) _data[i] -= a;
    return *this;
}

template <class T>
template <class U>
Tensor<1,T>& Tensor<1,T>::operator*=(const Tensor<1,U>& a)
{
    for (std::size_t i = 0; i < _data.size(); ++i) _data[i] *= a.vector()[i];
    return *this;
}

template <class T>
template <class U, typename std::enable_if<!std::is_same<Tensor<1,U>, typename std::decay<U>::type>::value, int>::type>
Tensor<1,T>& Tensor<1,T>::operator*=(const U& a)
{
    for (std::size_t i = 0; i < _data.size(); ++i) _data[i] *= a;
    return *this;
}

template <class T>
template <class U>
Tensor<1,T> Tensor<1,T>::operator+(const Tensor<1,U>& a) const
{
    Tensor<1,T> r;
    for (std::size_t i = 0; i < _data.size(); ++i) r._data.emplace_back(_data[i] + a.vector()[i]);
    return r;
}

template <class T>
template <class U, typename std::enable_if<!std::is_same<Tensor<1,U>, typename std::decay<U>::type>::value, int>::type>
Tensor<1,T> Tensor<1,T>::operator+(const U& a) const
{
    Tensor<1,T> r;
    for (std::size_t i = 0; i < _data.size(); ++i) r._data.emplace_back(_data[i] + a);
    return r;
}

template <class T>
template <class U>
Tensor<1,T> Tensor<1,T>::operator-(const Tensor<1,U>& a) const
{
    Tensor<1,T> r;
    for (std::size_t i = 0; i < _data.size(); ++i) r._data.emplace_back(_data[i] - a.vector()[i]);
    return r;
}

template <class T>
template <class U, typename std::enable_if<!std::is_same<Tensor<1,U>, typename std::decay<U>::type>::value, int>::type>
Tensor<1,T> Tensor<1,T>::operator-(const U& a) const
{
    Tensor<1,T> r;
    for (std::size_t i = 0; i < _data.size(); ++i) r._data.emplace_back(_data[i] - a);
    return r;
}

template <class T>
template <class U>
Tensor<1,T> Tensor<1,T>::operator*(const Tensor<1,U>& a) const
{
    Tensor<1,T> r;
    for (std::size_t i = 0; i < _data.size(); ++i) r._data.emplace_back(_data[i] * a.vector()[i]);
    return r;
}

template <class T>
template <class U, typename std::enable_if<!std::is_same<Tensor<1,U>, typename std::decay<U>::type>::value, int>::type>
Tensor<1,T> Tensor<1,T>::operator*(const U& a) const
{
    Tensor<1,T> r;
    for (std::size_t i = 0; i < _data.size(); ++i) r._data.emplace_back(_data[i] * a);
    return r;
}

template <class T>
bool Tensor<1,T>::operator==(const Tensor<1,T>& a) const
{
    return _data == a._data;
}

template <class T>
bool Tensor<1,T>::operator==(const T& a) const
{
    for (const auto& value : _data)
        if (value != a) return false;
    return true;
}

template <class T>
bool Tensor<1,T>::operator!=(const Tensor<1,T>& a) const
{
    return _data != a._data;
}

template <class T>
bool Tensor<1,T>::operator!=(const T& a) const
{
    return !(*this == a);
}

template <class T>
bool Tensor<1,T>::any_empty() const
{
    return empty();
}

template <class T>
T& Tensor<1,T>::back()
{
    return _data.back();
}

template <class T>
const T& Tensor<1,T>::back() const
{
    return _data.back();
}

template <class T>
typename std::vector<T>::iterator Tensor<1,T>::begin()
{
    return _data.begin();
}

template <class T>
typename std::vector<T>::const_iterator Tensor<1,T>::begin() const
{
    return _data.begin();
}

template <class T>
Tensor<1,T> Tensor<1,T>::copy(const std::vector<T>& data)
{
    Tensor<1,T> r;
    r._data = data;
    return r;
}

template <class T>
template <class U>
Tensor<1,T> Tensor<1,T>::copy(const std::vector<U>& data)
{
    Tensor<1,T> r;
    for (const auto& u : data) r._data.emplace_back(u);
    return r;
}

template <class T>
void Tensor<1,T>::emplace_back(const T& value)
{
    _data.emplace_back(value);
}

template <class T>
void Tensor<1,T>::emplace_back(T&& value)
{
    _data.emplace_back(std::move(value));
}

template <class T>
bool Tensor<1,T>::empty() const
{
    return _data.empty();
}

template <class T>
typename std::vector<T>::iterator Tensor<1,T>::end()
{
    return _data.end();
}

template <class T>
typename std::vector<T>::const_iterator Tensor<1,T>::end() const
{
    return _data.end();
}

template <class T>
T& Tensor<1,T>::front()
{
    return _data.front();
}

template <class T>
const T& Tensor<1,T>::front() const
{
    return _data.front();
}

template <class T>
bool Tensor<1,T>::front_empty() const
{
    return empty();
}

template <class T>
Tensor<1,T> Tensor<1,T>::move(std::vector<T>&& data)
{
    Tensor<1,T> r;
    r._data = std::move(data);
    return r;
}

template <class T>
template <class U>
Tensor<1,T> Tensor<1,T>::move(std::vector<U>&& data)
{
    Tensor<1,T> r;
    for (auto& u : data) r._data.emplace_back(std::move(u));
    return r;
}

template <class T>
void Tensor<1,T>::push_back(const T& value)
{
    _data.push_back(value);
}

template <class T>
void Tensor<1,T>::push_back(T&& value)
{
    _data.push_back(std::move(value));
}

template <class T>
void Tensor<1,T>::resize(std::size_t n)
{
    _data.resize(n);
}

template <class T>
void Tensor<1,T>::resize(std::size_t n, const T& value)
{
    _data.resize(n, value);
}

template <class T>
std::size_t Tensor<1,T>::size() const
{
    return _data.size();
}

template <class T>
std::vector<std::size_t> Tensor<1,T>::shape() const
{
    return {_data.size()};
}

template <class T>
std::vector<T>& Tensor<1,T>::vector()
{
    return _data;
}

template <class T>
const std::vector<T>& Tensor<1,T>::vector() const
{
    return _data;
}

} // type