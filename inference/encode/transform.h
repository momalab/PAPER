#pragma once

#include <cstddef>
#include <optional>
#include "tensor.h"

namespace fhe
{

template <int Dr, class Tr, int Di, class Ti>
type::Tensor<Dr,Tr> transform(const type::Tensor<Di,Ti>& t, std::size_t nthreads, const Ciphertext* reference);

template <int Dr, class Tr, int Di, class Ti>
type::Tensor<Dr,Tr> transform(const type::Tensor<Di,Ti>& t, std::size_t nthreads, int level = -1, double scale = 0.0);

} // fhe

#include "transform.hpp"