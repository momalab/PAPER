#pragma once

#include "scalar.h"
#include "seal/seal.h"

namespace fhe
{

using NativePlaintext = seal::Plaintext;
using NativeCiphertext = seal::Ciphertext;

} // fhe