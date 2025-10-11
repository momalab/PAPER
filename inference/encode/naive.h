#pragma once

#include <tuple>
#include "ciphertext.h"
#include "defines.h"
#include "plaintext.h"
#include "tensor.h"
#include "transform.h"

namespace fhe
{

namespace naive
{

inline type::Tensor<3,Scalar> decode_output(const type::Tensor<3,Ciphertext>& output) { return transform<3,Scalar,3,Ciphertext>(output, config::NTHREADS); }
inline type::Tensor<3,Scalar> decode_bias(const type::Tensor<3,Scalar>& bias) { return bias; }
inline type::Tensor<3,Scalar> decode_input(const type::Tensor<3,Ciphertext>& input) { return decode_output(input); }
inline type::Tensor<4,Scalar> decode_kernel(const type::Tensor<4,Scalar>& kernel) { return kernel; }

// Decode and decrypt output tensor
inline type::Tensor<3,Scalar> decode(const type::Tensor<3,Ciphertext>& output) { return decode_output(output); }

// Encode input, kernel, and bias
std::tuple
<
    type::Tensor<3,Ciphertext>, // Input: Ci x Ih x Iw
    type::Tensor<4,Scalar>, // Filter: Co x Ci x Fh x Fw
    type::Tensor<3,Scalar> // Bias: Co x Oh x Ow
>
encode
(
    const type::Tensor<3,Scalar>& input,
    const type::Tensor<4,Scalar>& kernel,
    const type::Tensor<3,Scalar>& bias
);

inline type::Tensor<3,Ciphertext> encode_input(const type::Tensor<3,Scalar>& input) { return transform<3,Ciphertext,3,Scalar>(input, config::NTHREADS); }
inline type::Tensor<3,Scalar> encode_bias(const type::Tensor<3,Scalar>& bias) { return bias; }
inline type::Tensor<4,Scalar> encode_kernel(const type::Tensor<4,Scalar>& kernel) { return kernel; }
inline type::Tensor<3,Ciphertext> encode_output(const type::Tensor<3,Scalar>& output) { return encode_input(output); }

} // naive

} // fhe