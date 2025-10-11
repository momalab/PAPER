#pragma once

#include <cstddef>
#include <vector>
#include <unordered_map>
#include "ciphertext.h"
#include "defines.h"
#include "hw.h"
#include "plaintext.h"
#include "tensor.h"

namespace fhe
{

namespace hw
{

type::Tensor<2,Ciphertext> // Co x nCT (Oh x Ow in Ih x Iw)
conv2d
(
    type::Tensor<2,Ciphertext>& input, // Ci x nCT (Ih x Iw)
    const type::Tensor<6,Scalar>& kernel, // Co x Kh x Kw x Ci x Ih x Iw
    const type::Tensor<3,Scalar>& bias, // Co x nCT x Ih x Iw (Oh x Ow in Ih x Iw)
    const Configuration& config,
    std::size_t nthreads = config::NTHREADS
);

type::Tensor<2,Ciphertext> // Co x nCT (Oh x Ow in Ih x Iw)
conv2d
(
    type::Tensor<2,Ciphertext>& input, // Ci x nCT (Ih x Iw)
    const type::Tensor<7,Scalar>& kernel, // (# models) x Co x Kh x Kw x Ci x Ih x Iw
    const type::Tensor<4,Scalar>& bias, // (# models) x Co x nCT x Ih x Iw (Oh x Ow in Ih x Iw)
    const Configuration& config,
    std::size_t nthreads,
    const std::vector<int>& offsets
);

} // hw

} // fhe