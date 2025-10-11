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

type::Tensor<2,Ciphertext> // Ci x nCT (Oh x Ow in Ih x Iw)
avgpool2d
(
    type::Tensor<2,Ciphertext>& input, // Ci x nCT (Ih x Iw)
    const Configuration& config,
    std::size_t nthreads = config::NTHREADS
);

} // hw

} // fhe