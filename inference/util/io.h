#pragma once

#include <ostream>
#include <string>
#include <vector>
#include "tensor.h"

namespace util
{

// print vectors
template <typename T> void print(const std::vector<T> & v, const std::string & offset = "");
template <typename T> std::string stringfy(const std::vector<T> & v, const std::string & offset = "");

// print tensors
template <int D, class T> void print(const type::Tensor<D,T> & t, const std::string & offset = "");
template <int D, class T> std::string stringfy(const type::Tensor<D,T> & t, const std::string & offset = "");

} // util

namespace std
{

// overload operator << for vectors and tensors
template <typename T> ostream& operator<<(ostream& os, const vector<T> & v);

} // std

namespace type
{

template <int D, class T> std::ostream& operator<<(std::ostream& os, const Tensor<D,T> & t);

} // type

#include "io.hpp"