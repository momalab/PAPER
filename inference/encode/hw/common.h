#pragma once

#include <vector>
#include "ciphertext.h"
#include "configuration.h"
#include "defines.h"
#include "plaintext.h"
#include "tensor.h"

namespace fhe
{

namespace hw
{

std::vector<bool> pattern_output(const Configuration& config);

type::Tensor<2,Ciphertext>
reformat
(
    const type::Tensor<2,Ciphertext>& output, // Co x nCT (Oh x Ow in Ih x Iw)
    const Configuration& config,
    const Scalar& scaling = 1.0
);

type::Tensor<2,fhe::Ciphertext>
remap
(
    const type::Tensor<2,Ciphertext>& input,
    const Configuration& input_config,
    const Configuration& output_config,
    const Scalar& scaling,
    double scale = 0.0,
    std::size_t nthreads = config::NTHREADS,
    const std::vector<int>& offsets = {0}
);

type::Tensor<4,fhe::Scalar> sanitize(const type::Tensor<4,fhe::Scalar>& kernel, const Configuration& config);

} // hw

} // fhe