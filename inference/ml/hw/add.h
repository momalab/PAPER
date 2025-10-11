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

type::Tensor<2,Ciphertext>
add
(
    type::Tensor<2,Ciphertext>& input1, // Ci x nCT (Ih1 x Iw1)
    type::Tensor<2,Ciphertext>& input2, // Ci x nCT (Ih2 x Iw2)
    const Configuration& config1,
    const Configuration& config2,
    std::size_t nthreads = config::NTHREADS,
    const std::vector<int>& offsets = {0}
);

void
add_inplace
(
    type::Tensor<2,Ciphertext>& input1, // Ci x nCT (Ih1 x Iw1)
    type::Tensor<2,Ciphertext>& input2, // Ci x nCT (Ih2 x Iw2)
    const Configuration& config1,
    const Configuration& config2,
    std::size_t nthreads = config::NTHREADS,
    const std::vector<int>& offsets = {0}
);

} // hw

} // fhe