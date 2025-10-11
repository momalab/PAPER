#pragma once

#include <cstddef>
#include <vector>
#include "ciphertext.h"
#include "hw.h"
#include "tensor.h"

namespace fhe
{

namespace hw
{

type::Tensor<3,Ciphertext> shift_input(const type::Tensor<2,Ciphertext>& input, const Configuration& config, std::size_t nthreads);

} // hw

} // fhe