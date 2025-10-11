#include "naive.h"

#include <tuple>
#include <utility>
#include "ciphertext.h"
#include "plaintext.h"
#include "tensor.h"
#include "vectorization.h"

using namespace std;
using namespace type;

namespace fhe
{

namespace naive
{

tuple
<
    Tensor<3,Ciphertext>, // Input: Ci x Ih x Iw
    Tensor<4,Scalar>, // Filter: Co x Ci x Fh x Fw
    Tensor<3,Scalar> // Bias: Co x Oh x Ow
>
encode
(
    const Tensor<3,Scalar>& input,
    const Tensor<4,Scalar>& kernel,
    const Tensor<3,Scalar>& bias
)
{
    auto x = encode_input(input);
    auto w = encode_kernel(kernel);
    auto b = encode_bias(bias);
    return make_tuple(move(x), move(w), move(b));
}

} // naive

} // fhe