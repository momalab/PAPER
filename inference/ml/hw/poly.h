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

type::Tensor<2,Ciphertext> // Ci x nCT (Ih x Iw)
poly
(
    type::Tensor<2,Ciphertext>& input, // Ci x nCT (Ih x Iw)
    const type::Tensor<4,Scalar>& kernel, // (n+1) x Ci x Ih x Iw
    const Configuration& config,
    std::size_t nthreads = config::NTHREADS
);

type::Tensor<2,Ciphertext> // Ci x nCT (Ih x Iw)
poly
(
    type::Tensor<2,Ciphertext>& input, // Ci x nCT (Ih x Iw)
    const type::Tensor<5,Scalar>& kernel, // (# models) x (n+1) x Ci x Ih x Iw
    const Configuration& config,
    std::size_t nthreads,
    const std::vector<int>& offsets
);

void // Ci x nCT (Ih x Iw)
poly_inplace
(
    type::Tensor<2,Ciphertext>& input, // Ci x nCT (Ih x Iw)
    const type::Tensor<4,Scalar>& kernel, // (n+1) x Ci x Ih x Iw
    const Configuration& config,
    std::size_t nthreads = config::NTHREADS
);

void // Ci x nCT (Ih x Iw)
poly_inplace
(
    type::Tensor<2,Ciphertext>& input, // Ci x nCT (Ih x Iw)
    const type::Tensor<5,Scalar>& kernel, // (# models) x (n+1) x Ci x Ih x Iw
    const Configuration& config,
    std::size_t nthreads,
    const std::vector<int>& offsets
);

} // hw

} // fhe