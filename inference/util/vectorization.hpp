#pragma once

#include <vector>

namespace util
{

// vector operations

template <class T, class U> std::vector<T> operator+=(std::vector<T> & a, const std::vector<U> & b)
{
    if (a.size() != b.size()) throw "Vector dimensions do not match";
    for (size_t i = 0; i < a.size(); ++i) a[i] += b[i];
    return a;
}

template <class T, class U> std::vector<T> operator-=(std::vector<T> & a, const std::vector<U> & b)
{
    if (a.size() != b.size()) throw "Vector dimensions do not match";
    for (size_t i = 0; i < a.size(); ++i) a[i] -= b[i];
    return a;
}

template <class T, class U> std::vector<T> operator*=(std::vector<T> & a, const std::vector<U> & b)
{
    if (a.size() != b.size()) throw "Vector dimensions do not match";
    for (size_t i = 0; i < a.size(); ++i) a[i] *= b[i];
    return a;
}

template <class T, class U> std::vector<T> operator+(const std::vector<T> & a, const std::vector<U> & b)
{
    if (a.size() != b.size()) throw "Vector dimensions do not match";
    std::vector<T> c(a.size());
    for (size_t i = 0; i < a.size(); ++i) c[i] = a[i] + b[i];
    return c;
}

template <class T, class U> std::vector<T> operator-(const std::vector<T> & a, const std::vector<U> & b)
{
    if (a.size() != b.size()) throw "Vector dimensions do not match";
    std::vector<T> c(a.size());
    for (size_t i = 0; i < a.size(); ++i) c[i] = a[i] - b[i];
    return c;
}

template <class T, class U> std::vector<T> operator*(const std::vector<T> & a, const std::vector<U> & b)
{
    if (a.size() != b.size()) throw "Vector dimensions do not match";
    std::vector<T> c(a.size());
    for (size_t i = 0; i < a.size(); ++i) c[i] = a[i] * b[i];
    return c;
}

// matrix operations

template <class T, class U> std::vector<std::vector<T>> operator+=(std::vector<std::vector<T>> & a, const std::vector<std::vector<U>> & b)
{
    if (a.size() != b.size() || a[0].size() != b[0].size()) throw "Matrix dimensions do not match";
    for (size_t i = 0; i < a.size(); ++i) a[i] += b[i];
    return a;
}

template <class T, class U> std::vector<std::vector<T>> operator-=(std::vector<std::vector<T>> & a, const std::vector<std::vector<U>> & b)
{
    if (a.size() != b.size() || a[0].size() != b[0].size()) throw "Matrix dimensions do not match";
    for (size_t i = 0; i < a.size(); ++i) a[i] -= b[i];
    return a;
}

template <class T, class U> std::vector<std::vector<T>> operator*=(std::vector<std::vector<T>> & a, const std::vector<std::vector<U>> & b)
{
    if (a.size() != b.size() || a[0].size() != b[0].size()) throw "Matrix dimensions do not match";
    for (size_t i = 0; i < a.size(); ++i) a[i] *= b[i];
    return a;
}

template <class T, class U> std::vector<std::vector<T>> operator+(const std::vector<std::vector<T>> & a, const std::vector<std::vector<U>> & b)
{
    if (a.size() != b.size() || a[0].size() != b[0].size()) throw "Matrix dimensions do not match";
    std::vector<std::vector<T>> c(a.size());
    for (size_t i = 0; i < a.size(); ++i) c[i] = a[i] + b[i];
    return c;
}

template <class T, class U> std::vector<std::vector<T>> operator-(const std::vector<std::vector<T>> & a, const std::vector<std::vector<U>> & b)
{
    if (a.size() != b.size() || a[0].size() != b[0].size()) throw "Matrix dimensions do not match";
    std::vector<std::vector<T>> c(a.size());
    for (size_t i = 0; i < a.size(); ++i) c[i] = a[i] - b[i];
    return c;
}

template <class T, class U> std::vector<std::vector<T>> operator*(const std::vector<std::vector<T>> & a, const std::vector<std::vector<U>> & b)
{
    if (a.size() != b.size() || a[0].size() != b[0].size()) throw "Matrix dimensions do not match";
    std::vector<std::vector<T>> c(a.size());
    for (size_t i = 0; i < a.size(); ++i) c[i] = a[i] * b[i];
    return c;
}

// tensor operations

template <class T, class U> std::vector<std::vector<std::vector<T>>>
operator+=(std::vector<std::vector<std::vector<T>>> & a, const std::vector<std::vector<std::vector<U>>> & b)
{
    if (a.size() != b.size() || a[0].size() != b[0].size() || a[0][0].size() != b[0][0].size()) throw "Tensor dimensions do not match";
    for (size_t i = 0; i < a.size(); ++i) a[i] += b[i];
    return a;
}

template <class T, class U> std::vector<std::vector<std::vector<T>>>
operator-=(std::vector<std::vector<std::vector<T>>> & a, const std::vector<std::vector<std::vector<U>>> & b)
{
    if (a.size() != b.size() || a[0].size() != b[0].size() || a[0][0].size() != b[0][0].size()) throw "Tensor dimensions do not match";
    for (size_t i = 0; i < a.size(); ++i) a[i] -= b[i];
    return a;
}

template <class T, class U> std::vector<std::vector<std::vector<T>>>
operator*=(std::vector<std::vector<std::vector<T>>> & a, const std::vector<std::vector<std::vector<U>>> & b)
{
    if (a.size() != b.size() || a[0].size() != b[0].size() || a[0][0].size() != b[0][0].size()) throw "Tensor dimensions do not match";
    for (size_t i = 0; i < a.size(); ++i) a[i] *= b[i];
    return a;
}

template <class T, class U> std::vector<std::vector<std::vector<T>>>
operator+(const std::vector<std::vector<std::vector<T>>> & a, const std::vector<std::vector<std::vector<U>>> & b)
{
    if (a.size() != b.size() || a[0].size() != b[0].size() || a[0][0].size() != b[0][0].size()) throw "Tensor dimensions do not match";
    std::vector<std::vector<std::vector<T>>> c(a.size());
    for (size_t i = 0; i < a.size(); ++i) c[i] = a[i] + b[i];
    return c;
}

template <class T, class U> std::vector<std::vector<std::vector<T>>>
operator-(const std::vector<std::vector<std::vector<T>>> & a, const std::vector<std::vector<std::vector<U>>> & b)
{
    if (a.size() != b.size() || a[0].size() != b[0].size() || a[0][0].size() != b[0][0].size()) throw "Tensor dimensions do not match";
    std::vector<std::vector<std::vector<T>>> c(a.size());
    for (size_t i = 0; i < a.size(); ++i) c[i] = a[i] - b[i];
    return c;
}

template <class T, class U> std::vector<std::vector<std::vector<T>>>
operator*(const std::vector<std::vector<std::vector<T>>> & a, const std::vector<std::vector<std::vector<U>>> & b)
{
    if (a.size() != b.size() || a[0].size() != b[0].size() || a[0][0].size() != b[0][0].size()) throw "Tensor dimensions do not match";
    std::vector<std::vector<std::vector<T>>> c(a.size());
    for (size_t i = 0; i < a.size(); ++i) c[i] = a[i] * b[i];
    return c;
}

// shape

template <class T> std::vector<std::size_t> shape(const std::vector<T> & v)
{
    return {v.size()};
}

template <class T> std::vector<std::size_t> shape(const std::vector<std::vector<T>> & v)
{
    return {v.size(), v[0].size()};
}

template <class T> std::vector<std::size_t> shape(const std::vector<std::vector<std::vector<T>>> & v)
{
    return {v.size(), v[0].size(), v[0][0].size()};
}

template <class T> std::vector<std::size_t> shape(const std::vector<std::vector<std::vector<std::vector<T>>>> & v)
{
    return {v.size(), v[0].size(), v[0][0].size(), v[0][0][0].size()};
}

// subvector

template <class T> std::vector<T> subvector(const std::vector<T> & v, std::size_t i, std::size_t j)
{
    if (!j || j > v.size()) j = v.size();
    if (i > j) throw "Invalid subvector range";
    return std::vector<T>(v.begin() + i, v.begin() + j);
}

} // util