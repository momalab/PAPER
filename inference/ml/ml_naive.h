#pragma once

#include <vector>
#include "ciphertext.h"
#include "plaintext.h"
#include "tensor.h"

namespace fhe
{

namespace naive
{

type::Tensor<3,Ciphertext> // Ci x Ih x Iw
add
(
    const type::Tensor<3,Ciphertext>& input1, // Ci x Ih x Iw
    const type::Tensor<3,Ciphertext>& input2  // Ci x Ih x Iw
);

type::Tensor<3,Ciphertext>
avgpool2d
(
    const type::Tensor<3,Ciphertext>& input, // Ci x Ih x Iw
    const std::vector<std::size_t>& kernel, // {Kh, Kw}
    const std::vector<std::size_t>& stride, // {Sh, Sw}
    const std::vector<std::size_t>& padding, // {Ph, Pw}
    const Scalar& divisor
);

type::Tensor<3,Ciphertext> // Ci x Ih x Iw
batchnorm2d
(
    const type::Tensor<3,Ciphertext>& input, // Ci x Ih x Iw
    const type::Tensor<4,Scalar>& kernel // (n+1) x Ci x Ih x Iw
);

type::Tensor<3,Ciphertext> // Co x Oh x Ow
conv2d
(
    const type::Tensor<3,Ciphertext>& input, // Ci x Ih x Iw
    const type::Tensor<6,Scalar>& kernel, // Co x Ci x Kh x Kw x Ih x Iw
    const type::Tensor<3,Scalar>& bias, // Co x Oh x Ow
    const std::vector<size_t>& stride, // {Sh, Sw}
    const std::vector<std::size_t>& padding // {Ph, Pw}
);

type::Tensor<3,Ciphertext> // 1 x 1 x Ko
linear
(
    const type::Tensor<3,Ciphertext>& input, // Ci x Ih x Iw
    const type::Tensor<4,Scalar>& kernel, // Ko x Ci x Ih x Iw
    const type::Tensor<3,Scalar>& bias // 1 x 1 x Ko
);

type::Tensor<3,Ciphertext> // Ci x Ih x Iw
poly
(
    const type::Tensor<3,Ciphertext>& input, // Ci x Ih x Iw
    const type::Tensor<4,Scalar>& kernel // (n+1) x Ci x Ih x Iw
);

type::Tensor<3,Ciphertext> // Ci x Ih x Iw
polyskip
(
    const type::Tensor<3,Ciphertext>& input1, // Ci x Ih x Iw
    const type::Tensor<3,Ciphertext>& input2, // Ci x Ih x Iw
    const type::Tensor<4,Scalar>& kernel // 2n x Ci x Ih x Iw
);

} // naive
    
} // fhe