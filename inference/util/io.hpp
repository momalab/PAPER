#pragma once

#include <iomanip>
#include <iostream>
#include <ostream>
#include <string>
#include <type_traits>
#include <vector>
#include "tensor.h"

namespace util
{

// prevent ambiguous overload of std::vector<T> and std::vector<std::vector<T>>
template <typename T> struct is_vector : std::false_type {};
template <typename T> struct is_vector<std::vector<T>> : std::true_type {};

// print vectors
template <typename T>
std::string stringfy(const std::vector<T> & v, const std::string & offset)
{
    std::ostringstream os;
    if constexpr (is_vector<T>::value)
    {
        os << offset << "{\n";
        for (const auto & r : v) os << stringfy(r, offset+" ") << '\n';
        os << offset << '}';
    }
    else
    {
        // os << std::fixed << std::setprecision(6);
        os << std::scientific << std::setprecision(6);
        os << std::showpos;
        os << offset << "{ ";
        for (const auto & e : v) os << e << ' ';
        os << '}';
    }
    return os.str();
}

template <typename T>
void print(const std::vector<T> & v, const std::string & offset)
{
    std::cout << stringfy(v, offset);
}

// print tensors
template <int D, class T>
std::string stringfy(const type::Tensor<D,T> & t, const std::string & offset)
{
    std::ostringstream os;
    if constexpr (D == 1) os << stringfy(t.vector(), offset);
    else
    {
        os << offset << "{\n";
        for (const auto & e : t) os << stringfy(e, offset+" ") << '\n';
        os << offset << '}';
    }
    return os.str();
}

template <int D, class T> void print(const type::Tensor<D,T> & t, const std::string & offset)
{
    std::cout << stringfy(t, offset);
}

} // util

namespace std
{

template <typename T> ostream& operator<<(ostream& os, const vector<T> & v)
{
    return os << util::stringfy(v);
}

} // std

namespace type
{

template <int D, class T> std::ostream& operator<<(std::ostream& os, const Tensor<D,T> & t)
{
    return os << util::stringfy(t);
}

} // type