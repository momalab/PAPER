#pragma once

#include <cstddef>
#include <vector>
#include "ciphertext.h"
#include "defines.h"
#include "hw.h"
#include "plaintext.h"
#include "tensor.h"

namespace fhe
{

namespace hw
{

type::Tensor<2,Ciphertext> // 1 x nCTo (Ko x Ih x Iw) // each ko uses Ih x Iw slots
linear
(
    type::Tensor<2,Ciphertext>& input, // Ci x nCT (Ih x Iw)
    const type::Tensor<4,Scalar>& kernel, // Ko x Ci x Ih x Iw
    const type::Tensor<3,Scalar>& bias, // 1 x 1 x Ko
    const Configuration& config,
    std::size_t nthreads = config::NTHREADS
);

type::Tensor<2,Ciphertext> // 1 x nCTo (Ko x Ih x Iw) // each ko uses Ih x Iw slots
linear
(
    type::Tensor<2,Ciphertext>& input, // Ci x nCT (Ih x Iw)
    const type::Tensor<5,Scalar>& kernel, // (# models) x Ko x Ci x Ih x Iw
    const type::Tensor<4,Scalar>& bias, // (# models) x 1 x 1 x Ko
    const Configuration& config,
    std::size_t nthreads,
    const std::vector<int>& offsets
);

} // hw

} // fhe