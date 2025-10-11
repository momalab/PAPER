#pragma once

#include <vector>

namespace util
{

// vector operations

template <class T, class U> std::vector<T> operator+=(std::vector<T> & a, const std::vector<U> & b);
template <class T, class U> std::vector<T> operator-=(std::vector<T> & a, const std::vector<U> & b);
template <class T, class U> std::vector<T> operator*=(std::vector<T> & a, const std::vector<U> & b);
template <class T, class U> std::vector<T> operator+(const std::vector<T> & a, const std::vector<U> & b);
template <class T, class U> std::vector<T> operator-(const std::vector<T> & a, const std::vector<U> & b);
template <class T, class U> std::vector<T> operator*(const std::vector<T> & a, const std::vector<U> & b);

// matrix operations

template <class T, class U> std::vector<std::vector<T>> operator+=(std::vector<std::vector<T>> & a, const std::vector<std::vector<U>> & b);
template <class T, class U> std::vector<std::vector<T>> operator-=(std::vector<std::vector<T>> & a, const std::vector<std::vector<U>> & b);
template <class T, class U> std::vector<std::vector<T>> operator*=(std::vector<std::vector<T>> & a, const std::vector<std::vector<U>> & b);
template <class T, class U> std::vector<std::vector<T>> operator+(const std::vector<std::vector<T>> & a, const std::vector<std::vector<U>> & b);
template <class T, class U> std::vector<std::vector<T>> operator-(const std::vector<std::vector<T>> & a, const std::vector<std::vector<U>> & b);
template <class T, class U> std::vector<std::vector<T>> operator*(const std::vector<std::vector<T>> & a, const std::vector<std::vector<U>> & b);

// tensor operations

template <class T, class U> std::vector<std::vector<std::vector<T>>>
operator+=(std::vector<std::vector<std::vector<T>>> & a, const std::vector<std::vector<std::vector<U>>> & b);

template <class T, class U> std::vector<std::vector<std::vector<T>>>
operator+=(std::vector<std::vector<std::vector<T>>> & a, const std::vector<std::vector<std::vector<U>>> & b);

template <class T, class U> std::vector<std::vector<std::vector<T>>>
operator*=(std::vector<std::vector<std::vector<T>>> & a, const std::vector<std::vector<std::vector<U>>> & b);

template <class T, class U> std::vector<std::vector<std::vector<T>>>
operator+(const std::vector<std::vector<std::vector<T>>> & a, const std::vector<std::vector<std::vector<U>>> & b);

template <class T, class U> std::vector<std::vector<std::vector<T>>>
operator-(const std::vector<std::vector<std::vector<T>>> & a, const std::vector<std::vector<std::vector<U>>> & b);

template <class T, class U> std::vector<std::vector<std::vector<T>>>
operator*(const std::vector<std::vector<std::vector<T>>> & a, const std::vector<std::vector<std::vector<U>>> & b);

// shape

template <class T> std::vector<std::size_t> shape(const std::vector<T> & v);
template <class T> std::vector<std::size_t> shape(const std::vector<std::vector<T>> & v);
template <class T> std::vector<std::size_t> shape(const std::vector<std::vector<std::vector<T>>> & v);
template <class T> std::vector<std::size_t> shape(const std::vector<std::vector<std::vector<std::vector<T>>>> & v);

// subvector

template <class T> std::vector<T> subvector(const std::vector<T> & v, std::size_t i, std::size_t j=0);

} // util

#include "vectorization.hpp"