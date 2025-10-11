#include "ml_plain.h"

#include <stdexcept>
#include <utility>
#include <vector>
#include "native.h"
#include "tensor.h"

using namespace std;
using namespace type;

namespace fhe
{

namespace plain
{

Tensor<3,Scalar> // Ci x Ih x Iw
add
(
    const Tensor<3,Scalar>& input1, // Ci x Ih x Iw
    const Tensor<3,Scalar>& input2  // Ci x Ih x Iw
)
{
    return input1 + input2;
}

Tensor<3,Scalar>
avgpool2d
(
    const Tensor<3,Scalar>& input, // Ci x Ih x Iw
    const vector<size_t>& kernel, // {Kh, Kw}
    const vector<size_t>& stride, // {Sh, Sw}
    const vector<size_t>& padding, // {Ph, Pw}
    const Scalar& divisor
)
{
    if (input.front_empty()) throw invalid_argument("fhe::plain::avgpool2d: invalid input");
    if (kernel.size() != 4) throw "fhe::plain::avgpool2d: Kernel must be a 4-element vector {1, 1, Kh, Kw}";
    if (stride.size() != 2) throw "fhe::plain::avgpool2d: Stride must be a 2-element vector {Sh, Sw}";
    if (padding.size() != 2) throw "fhe::plain::avgpool2d: Padding must be a 2-element vector {Ph, Pw}";

    const auto& ci = input.size();
    const auto& ih = input[0].size();
    const auto& iw = input[0][0].size();
    const auto& kh = kernel[2];
    const auto& kw = kernel[3];
    const auto& sh = stride[0];
    const auto& sw = stride[1];
    const auto& ph = padding[0];
    const auto& pw = padding[1];
    const auto& oh = (ih + 2*ph - kh) / sh + 1;
    const auto& ow = (iw + 2*pw - kw) / sw + 1;

    auto x = pad(input, padding);
    bool divide = abs(divisor - 1.0) > 1e-8;
    const Scalar multiplier = 1.0 / divisor;
    Tensor<3,Scalar> output{ci, oh, ow};
    for (size_t i = 0; i < ci; ++i)
    {
        for (size_t h = 0; h < oh; ++h)
        {
            for (size_t w = 0; w < ow; ++w)
            {
                Scalar sum = 0;
                for (size_t j = 0; j < kh; ++j)
                    for (size_t k = 0; k < kw; ++k)
                        sum += x[i][h * sh + j][w * sw + k];
                output[i][h][w] = move(sum); // equivalent to divisor_override = 1 (instead of the default: kh * kw)
                if (divide) output[i][h][w] *= multiplier;
            }
        }
    }
    return output;
}

Tensor<3,Scalar> // Ci x Ih x Iw
batchnorm2d
(
    const Tensor<3,Scalar>& input, // Ci x Ih x Iw
    const Tensor<4,Scalar>& kernel // (n+1) x Ci x Ih x Iw
)
{
    const auto& ksize = kernel.size();
    if (ksize != 2 && ksize != 1) throw "fhe::plain::batchnorm: Kernel must be a 1 or 2-element vector";
    if (ksize == 2) return input * kernel[1] + kernel[0];
    return input + kernel[0];
}

Tensor<3,Scalar> // Co x Oh x Ow
conv2d
(
    const Tensor<3,Scalar>& input, // Ci x Ih x Iw
    const Tensor<6,Scalar>& kernel, // Co x Kh x Kw x Ci x Ih x Iw
    const Tensor<3,Scalar>& bias, // Co x Oh x Ow
    const vector<size_t>& stride, // {Sh, Sw}
    const vector<size_t>& padding // {Ph, Pw}
)
{
    if (input.front_empty()) throw invalid_argument("fhe::plain::conv2d: invalid input");
    if (kernel.front_empty()) throw invalid_argument("fhe::plain::conv2d: invalid kernel");
    if (bias.front_empty()) throw invalid_argument("fhe::plain::conv2d: invalid bias");
    if (stride.size() != 2) throw "fhe::plain::conv2d: Stride must be a 2-element vector";
    if (input.size() != kernel[0][0][0].size()) throw "fhe::plain::conv2d: Input channels in input and kernel do not match";
    if (kernel.size() != bias.size()) throw "fhe::plain::conv2d: Output channels in kernel and bias do not match";
    if (padding.size() != 2) throw "fhe::plain::avgpool2d: Padding must be a 2-element vector {Ph, Pw}";

    const auto& ci = input.size();
    const auto& ih = input[0].size();
    const auto& iw = input[0][0].size();
    const auto& kh = kernel[0].size();
    const auto& kw = kernel[0][0].size();
    const auto& sh = stride[0];
    const auto& sw = stride[1];
    const auto& ph = padding[0];
    const auto& pw = padding[1];
    const auto& co = kernel.size();
    const auto& oh = (ih + 2*ph - kh) / sh + 1;
    const auto& ow = (iw + 2*pw - kw) / sw + 1;

    if (bias[0].size() != oh || bias[0][0].size() != ow) throw "fhe::plain::conv2d: Bias dimensions do not match output dimensions";

    const auto& x = input;
    Tensor<3,Scalar> output{co, oh, ow};
    for (size_t o = 0; o < co; ++o)
    {
        for (size_t h = 0; h < oh; ++h)
        {
            for (size_t w = 0; w < ow; ++w)
            {
                output[o][h][w] = bias[o][h][w];
                for (size_t i = 0; i < ci; ++i)
                {
                    for (size_t krow = 0; krow < kh; ++krow)
                    {
                        for (size_t kcol = 0; kcol < kw; ++kcol)
                        {
                            size_t idx = h * sh + krow - ph;
                            size_t jdx = w * sw + kcol - pw;
                            if (idx >= ih || jdx >= iw) continue;
                            output[o][h][w] += x[i][idx][jdx] * kernel[o][krow][kcol][i][idx][jdx];
                        }
                    }
                }
            }
        }
    }
    return output;
}

Tensor<3,Scalar> // 1 x 1 x Ko
linear
(
    const Tensor<3,Scalar>& input, // Ci x Ih x Iw
    const Tensor<4,Scalar>& kernel, // Ko x Ci x Ih x Iw
    const Tensor<3,Scalar>& bias // 1 x 1 x Ko
)
{
    if (input.front_empty()) throw invalid_argument("fhe::plain::linear: invalid input");
    if (kernel.front_empty()) throw invalid_argument("fhe::plain::linear: invalid kernel");
    if (bias.front_empty()) throw invalid_argument("fhe::plain::linear: invalid bias");
    if (kernel[0][0].size() != input[0].size() || kernel[0][0][0].size() != input[0][0].size()) throw "fhe::plain::linear: Kernel dimensions do not match input dimensions";
    if (kernel.size() != bias.size()) throw "fhe::plain::linear: Output channels in kernel and bias do not match";

    const auto& ci = input.size();
    const auto& ih = input[0].size();
    const auto& iw = input[0][0].size();
    const auto& ko = kernel.size();

    Tensor<3,Scalar> output{ko, 1, 1};
    for (size_t o = 0; o < ko; ++o)
    {
        Scalar sum = bias[o][0][0];
        for (size_t i = 0; i < ci; ++i)
        {
            for (size_t ii = 0; ii < ih; ++ii)
            {
                for (size_t ij = 0; ij < iw; ++ij)
                {
                    sum += input[i][ii][ij] * kernel[o][i][ii][ij];
                }
            }
        }
        output[o][0][0] = sum;
    }
    return output;
}

Tensor<3,Scalar> // Ci x Ih x Iw
offset
(
    const Tensor<3,Scalar>& input, // Ci x Ih x Iw
    const Tensor<3,Scalar>& bias  // Ci x Ih x Iw
)
{
    return input + bias;
}

Tensor<3,Scalar> // Ci x (Ih + Ph) x (Iw + Pw)
pad
(
    const Tensor<3,Scalar>& input, // Ci x Ih x Iw
    const vector<size_t>& padding // {Ph, Pw}
)
{
    if (padding.size() != 2) throw "fhe::plain::pad: Padding must be a 2-element vector {Ph, Pw}";
    const auto& ph = padding[0];
    const auto& pw = padding[1];
    if (ph == 0 && pw == 0) return input;

    const auto& ci = input.size();
    const auto& ih = input[0].size();
    const auto& iw = input[0][0].size();
    const auto& oh = ih + 2*ph;
    const auto& ow = iw + 2*pw;

    Tensor<3,Scalar> output{ci, oh, ow};
    for (size_t i = 0; i < ci; ++i)
        for (size_t h = 0; h < ih; ++h)
            for (size_t w = 0; w < iw; ++w)
                output[i][h + ph][w + pw] = input[i][h][w];
    return output;
}

Tensor<3,Scalar> // Ci x Ih x Iw
poly
(
    const Tensor<3,Scalar>& input, // Ci x Ih x Iw
    const Tensor<1,Scalar>& kernel // (n+1)
)
{
    const auto& ksize = kernel.size();
    if (ksize != 3 && ksize != 2) throw "fhe::plain::poly: Kernel must be at least a 2 or 3-element vector";
    if (ksize == 3) return (input * input) * kernel[2] + input * kernel[1] + kernel[0];
    return (input * input) + input * kernel[1] + kernel[0];
}

Tensor<3,Scalar> // Ci x Ih x Iw
poly
(
    const Tensor<3,Scalar>& input, // Ci x Ih x Iw
    const Tensor<4,Scalar>& kernel // (n+1) x Ci x Ih x Iw
)
{
    const auto& ksize = kernel.size();
    if (ksize != 3 && ksize != 2) throw "fhe::plain::poly: Kernel must be at least a 2 or 3-element vector";
    if (ksize == 3) return (input * input) * kernel[2] + input * kernel[1] + kernel[0];
    return (input * input) + input * kernel[1] + kernel[0];
}

Tensor<3,Scalar> // Ci x Ih x Iw
polylow
(
    const Tensor<3,Scalar>& input, // Ci x Ih x Iw
    const Tensor<4,Scalar>& kernel // n x Ci x Ih x Iw
)
{
    if (kernel.size() != 2) throw "fhe::plain::poly: Kernel must be at least a 2-element vector";
    return (input * input) + input * kernel[1] + kernel[0];
}

Tensor<3,Scalar> // Ci x Ih x Iw
polyskip
(
    const Tensor<3,Scalar>& input1, // Ci x Ih x Iw
    const Tensor<3,Scalar>& input2, // Ci x Ih x Iw
    const Tensor<4,Scalar>& kernel // 2n x Ci x Ih x Iw
)
{
    const auto& ksize = kernel.size();
    if (ksize != 6 && ksize != 5) throw "fhe::plain::polyskip: Kernel must be at least a 5 or 6-element vector";
    const auto& xa = input1;
    const auto& xb = input2;
    auto xa2 = xa * xa;
    auto xb2 = xb * xb;
    const auto& d0 = kernel[0];
    const auto& db = kernel[1];
    const auto& da = kernel[2];
    const auto& dab= kernel[3];
    const auto& db2= kernel[4];
    if (ksize == 5) return xa2 + xb2*db2 + xa*(xb*dab) + xa*da + xb*db + d0;
    const auto& da2= kernel[5];
    return xa2*da2 + xb2*db2 + xa*(xb*dab) + xa*da + xb*db + d0;
}

Tensor<3,Scalar> // Ci x Ih x Iw
polyskiplow
(
    const Tensor<3,Scalar>& input1, // Ci x Ih x Iw
    const Tensor<3,Scalar>& input2, // Ci x Ih x Iw
    const Tensor<4,Scalar>& kernel // (2n-1) x Ci x Ih x Iw
)
{
    if (kernel.size() != 5) throw "fhe::plain::polyskip: Kernel must be at least a 5-element vector";
    const auto& xa = input1;
    const auto& xb = input2;
    auto xa2 = xa * xa;
    auto xb2 = xb * xb;
    const auto& d0 = kernel[0];
    const auto& da = kernel[1];
    const auto& db = kernel[2];
    const auto& db2 = kernel[3];
    const auto& dab = kernel[4];
    return xa2 + xb2*db2 + xa*(xb*dab) + xa*da + xb*db + d0;
}

} // plain

} // fhe