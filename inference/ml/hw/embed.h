#pragma once

#include <vector>
#include "adapt.h"
#include "ciphertext.h"
#include "hw.h"
#include "plaintext.h"
#include "tensor.h"

namespace fhe
{

namespace hw
{

type::Tensor<2,Ciphertext> // Co x nCT (Oh x Ow)
avgpool2d_reformat
(
    const type::Tensor<2,Ciphertext>& input, // Ci x nCT (Ih x Iw)
    const Configuration& config
);

type::Tensor<2,Ciphertext> // Co x nCT (Oh x Ow)
conv2d_reformat
(
    const type::Tensor<2,Ciphertext>& input, // Ci x nCT (Ih x Iw)
    const type::Tensor<5,Plaintext>& kernel, // Co x Ci x Kh x Kw x nCT
    const type::Tensor<2,Plaintext>& bias, // Co x nCT (Oh x Ow)
    const Configuration& config
);

type::Tensor<2,Ciphertext> // 1 x nCTo (1 x Ko) // each ko uses 1 slot
linear_reformat
(
    const type::Tensor<2,Ciphertext>& input, // Ci x nCT (Ih x Iw)
    const type::Tensor<3,Plaintext>& kernel, // Ko x Ci x nCT (Ih x Iw)
    const type::Tensor<2,Plaintext>& bias, // 1 x nCTo (1 x Ko)
    const Configuration& config
);

} // hw

} // fhe