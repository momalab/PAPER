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
polyskip
(
    type::Tensor<2,Ciphertext>& input1, // Ci x nCT (Ih x Iw)
    type::Tensor<2,Ciphertext>& input2, // Ci x nCT (Ih x Iw)
    const type::Tensor<4,Scalar>& kernel, // 2n x Ci x nCT x Ih x Iw
    const Configuration& config1,
    const Configuration& config2,
    std::size_t nthreads = config::NTHREADS
);

void // Ci x nCT (Ih x Iw)
polyskip_inplace
(
    type::Tensor<2,Ciphertext>& input1, // Ci x nCT (Ih x Iw)
    type::Tensor<2,Ciphertext>& input2, // Ci x nCT (Ih x Iw)
    const type::Tensor<4,Scalar>& kernel, // 2n x Ci x nCT x Ih x Iw
    const Configuration& config1,
    const Configuration& config2,
    std::size_t nthreads = config::NTHREADS
);

} // hw

} // fhe