#pragma once

#include <vector>
#include "native.h"
#include "tensor.h"

namespace fhe
{

namespace plain
{

type::Tensor<3,Scalar> // Ci x Ih x Iw
add
(
    const type::Tensor<3,Scalar>& input1, // Ci x Ih x Iw
    const type::Tensor<3,Scalar>& input2  // Ci x Ih x Iw
);

type::Tensor<3,Scalar> // Ci x Oh x Ow
avgpool2d
(
    const type::Tensor<3,Scalar>& input, // Ci x Ih x Iw
    const std::vector<std::size_t>& kernel, // {Kh, Kw}
    const std::vector<std::size_t>& stride, // {Sh, Sw}
    const std::vector<std::size_t>& padding, // {Ph, Pw}
    const Scalar& divisor
);

type::Tensor<3,Scalar> // Ci x Ih x Iw
batchnorm2d
(
    const type::Tensor<3,Scalar>& input, // Ci x Ih x Iw
    const type::Tensor<4,Scalar>& kernel // (n+1) x Ci x Ih x Iw
);

type::Tensor<3,Scalar> // Co x Oh x Ow
conv2d
(
    const type::Tensor<3,Scalar>& input, // Ci x Ih x Iw
    const type::Tensor<6,Scalar>& kernel, // Co x Kh x Kw x Ci x Ih x Iw
    const type::Tensor<3,Scalar>& bias, // Co x Oh x Ow
    const std::vector<std::size_t>& stride, // {Sh, Sw}
    const std::vector<std::size_t>& padding // {Ph, Pw}
);

type::Tensor<3,Scalar> // 1 x 1 x Ko
linear
(
    const type::Tensor<3,Scalar>& input, // Ci x Ih x Iw
    const type::Tensor<4,Scalar>& kernel, // Ko x Ci x Ih x Iw
    const type::Tensor<3,Scalar>& bias // 1 x 1 x Ko
);

type::Tensor<3,Scalar> // Ci x Ih x Iw
offset
(
    const type::Tensor<3,Scalar>& input, // Ci x Ih x Iw
    const type::Tensor<3,Scalar>& bias  // Ci x Ih x Iw
);

type::Tensor<3,Scalar> // Ci x (Ih + Ph) x (Iw x Pw)
pad
(
    const type::Tensor<3,Scalar>& input, // Ci x Ih x Iw
    const std::vector<std::size_t>& padding // {Ph, Pw}
);

type::Tensor<3,Scalar> // Ci x Ih x Iw
poly
(
    const type::Tensor<3,Scalar>& input, // Ci x Ih x Iw
    const type::Tensor<1,Scalar>& kernel // (n+1)
);

type::Tensor<3,Scalar> // Ci x Ih x Iw
poly
(
    const type::Tensor<3,Scalar>& input, // Ci x Ih x Iw
    const type::Tensor<4,Scalar>& kernel // (n+1) x Ci x Ih x Iw
);

type::Tensor<3,Scalar> // Ci x Ih x Iw
polylow
(
    const type::Tensor<3,Scalar>& input, // Ci x Ih x Iw
    const type::Tensor<4,Scalar>& kernel // n x Ci x Ih x Iw
);

type::Tensor<3,Scalar> // Ci x Ih x Iw
polyskip
(
    const type::Tensor<3,Scalar>& input1, // Ci x Ih x Iw
    const type::Tensor<3,Scalar>& input2, // Ci x Ih x Iw
    const type::Tensor<4,Scalar>& kernel // 2n x Ci x Ih x Iw
);

type::Tensor<3,Scalar> // Ci x Ih x Iw
polyskiplow
(
    const type::Tensor<3,Scalar>& input1, // Ci x Ih x Iw
    const type::Tensor<3,Scalar>& input2, // Ci x Ih x Iw
    const type::Tensor<4,Scalar>& kernel // (2n-1) x Ci x Ih x Iw
);

} // plain

} // fhe