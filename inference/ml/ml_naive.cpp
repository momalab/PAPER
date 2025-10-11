#include "ml_naive.h"

#include <stdexcept>
#include <vector>
#include <utility>
#include "ciphertext.h"
#include "plaintext.h"
#include "tensor.h"

using namespace std;
using namespace type;

namespace fhe
{

namespace naive
{

Tensor<3,Ciphertext> // Ci x Ih x Iw
add
(
    const Tensor<3,Ciphertext>& input1, // Ci x Ih x Iw
    const Tensor<3,Ciphertext>& input2  // Ci x Ih x Iw
)
{
    return input1 + input2;
}

Tensor<3,Ciphertext>
avgpool2d
(
    const Tensor<3,Ciphertext>& input, // Ci x Ih x Iw
    const vector<size_t>& kernel, // {Kh, Kw}
    const vector<size_t>& stride, // {Sh, Sw}
    const vector<size_t>& padding, // {Ph, Pw}
    const Scalar& divisor
)
{
    cout << "u\n";
    if (input.front_empty()) throw invalid_argument("fhe::naive::avgpool2d: invalid input");
    if (kernel.size() != 4) throw "fhe::naive::avgpool2d: Kernel must be a 4-element vector {1, 1, Kh, Kw}";
    if (stride.size() != 2) throw "fhe::naive::avgpool2d: Stride must be a 2-element vector {Sh, Sw}";
    if (padding.size() != 2) throw "fhe::naive::avgpool2d: Padding must be a 2-element vector {Ph, Pw}";

    const auto& kh = kernel[2];
    const auto& kw = kernel[3];
    const auto& sh = stride[0];
    const auto& sw = stride[1];
    const auto& ci = input.size();
    const auto& ih = input[0].size();
    const auto& iw = input[0][0].size();
    const auto& ph = padding[0];
    const auto& pw = padding[1];
    const auto& oh = (ih + 2*ph - kh) / sh + 1;
    const auto& ow = (iw + 2*pw - kw) / sw + 1;

    bool divide = abs(divisor - 1.0) > 1e-8;
    const Scalar multiplier = 1.0 / divisor;
    Tensor<3,Ciphertext> output{ci, oh, ow};
    for (size_t i = 0; i < ci; i++)
    {
        for (size_t h = 0; h < oh; h++)
        {
            for (size_t w = 0; w < ow; w++)
            {
                vector<Ciphertext> v;
                for (size_t j = 0; j < kh; j++)
                {
                    for (size_t k = 0; k < kw; k++)
                    {
                        size_t idx = h * sh + j - ph;
                        size_t jdx = w * sw + k - pw;
                        if (idx >= ih || jdx >= iw) continue;
                        v.emplace_back(input[i][idx][jdx]);
                    }
                }
                output[i][h][w] = move(Ciphertext::add_inplace(v));
                if (divide) output[i][h][w] *= multiplier;
            }
        }
    }
    return output;
}

Tensor<3,Ciphertext> // Ci x Ih x Iw
batchnorm2d
(
    const Tensor<3,Ciphertext>& input, // Ci x Ih x Iw
    const Tensor<4,Scalar>& kernel // (n+1) x Ci x Ih x Iw
)
{
    const auto& ksize = kernel.size();
    if (ksize != 2 && ksize != 1) throw "fhe::naive::batchnorm: Kernel must be a 1 or 2-element vector";
    // if (ksize == 2) return input * kernel[1] + kernel[0];
    if (ksize == 2)
    {
        auto r = input * kernel[1];
        Ciphertext::refit_inplace(r);
        r += kernel[0];
        return r;
    }
    return input + kernel[0];
}

Tensor<3,Ciphertext> // Co x Oh x Ow
conv2d
(
    const Tensor<3,Ciphertext>& input, // Ci x Ih x Iw
    const Tensor<6,Scalar>& kernel, // Co x Ci x Kh x Kw x Ih x Iw
    const Tensor<3,Scalar>& bias, // Co x Oh x Ow
    const vector<size_t>& stride, // {Sh, Sw}
    const vector<size_t>& padding // {Ph, Pw}
)
{
    if (input.front_empty()) throw invalid_argument("input");
    if (kernel.front_empty()) throw invalid_argument("kernel");
    if (bias.front_empty()) throw invalid_argument("bias");
    if (stride.size() != 2) throw "fhe::naive::conv2d: Stride must be a 2-element vector";
    if (input.size() != kernel[0][0][0].size()) throw "fhe::naive::conv2d: Input channels in input and kernel do not match";
    if (kernel.size() != bias.size()) throw "fhe::naive::conv2d: Output channels in kernel and bias do not match";
    if (padding.size() != 2) throw "fhe::naive::avgpool2d: Padding must be a 2-element vector {Ph, Pw}";

    const auto& sh = stride[0];
    const auto& sw = stride[1];
    const auto& ci = input.size();
    const auto& ih = input[0].size();
    const auto& iw = input[0][0].size();
    const auto& co = kernel.size();
    const auto& kh = kernel[0].size();
    const auto& kw = kernel[0][0].size();
    const auto& ph = padding[0];
    const auto& pw = padding[1];
    const auto& oh = (ih + 2*ph - kh) / sh + 1;
    const auto& ow = (iw + 2*pw - kw) / sw + 1;

    if (bias[0].size() != oh || bias[0][0].size() != ow) throw "fhe::naive::conv2d: Bias dimensions do not match output dimensions";

    Tensor<3,Ciphertext> output{co, oh, ow};
    for (size_t o = 0; o < co; o++)
    {
        for (size_t h = 0; h < oh; h++)
        {
            for (size_t w = 0; w < ow; w++)
            {
                vector<Ciphertext> v;
                for (size_t i = 0; i < ci; i++)
                {
                    for (size_t j = 0; j < kh; j++)
                    {
                        for (size_t k = 0; k < kw; k++)
                        {
                            size_t idx = h * sh + j - ph;
                            size_t jdx = w * sw + k - pw;
                            if (idx >= ih || jdx >= iw) continue;
                            v.emplace_back(input[i][idx][jdx] * kernel[o][j][k][i][idx][jdx]);
                        }
                    }
                }
                output[o][h][w] = move(Ciphertext::add_inplace(v));
                output[o][h][w].refit_inplace();
                output[o][h][w] += bias[o][h][w];
            }
        }
    }
    return output;
}

Tensor<3,Ciphertext> // 1 x 1 x Ko
linear
(
    const Tensor<3,Ciphertext>& input, // Ci x Ih x Iw
    const Tensor<4,Scalar>& kernel, // Ko x Ci x Ih x Iw
    const Tensor<3,Scalar>& bias // 1 x 1 x Ko
)
{
    if (input.front_empty()) throw invalid_argument("fhe::naive::linear: invalid input");
    if (kernel.front_empty()) throw invalid_argument("fhe::naive::linear: invalid kernel");
    if (bias.front_empty()) throw invalid_argument("fhe::naive::linear: invalid bias");
    if (kernel[0][0].size() != input[0].size() || kernel[0][0][0].size() != input[0][0].size()) throw "fhe::naive::linear: Kernel dimensions do not match input dimensions";
    if (kernel.size() != bias.size()) throw "fhe::naive::linear: Output channels in kernel and bias do not match";

    const auto& ci = input.size();
    const auto& ih = input[0].size();
    const auto& iw = input[0][0].size();
    const auto& ko = kernel.size();

    Tensor<3,Ciphertext> output{ko, 1, 1};
    for (size_t o = 0; o < ko; ++o)
    {
        Ciphertext& sum = output[o][0][0];
        for (size_t i = 0; i < ci; ++i)
        {
            for (size_t ii = 0; ii < ih; ++ii)
            {
                for (size_t ij = 0; ij < iw; ++ij)
                {
                    if (!i && !ii && !ij) sum = input[i][ii][ij] * kernel[o][i][ii][ij];
                    else sum += input[i][ii][ij] * kernel[o][i][ii][ij];
                }
            }
        }
        sum.refit_inplace();
        sum += bias[o][0][0];
    }
    return output;
}

Tensor<3,Ciphertext> // Ci x Ih x Iw
poly
(
    const Tensor<3,Ciphertext>& input, // Ci x Ih x Iw
    const Tensor<4,Scalar>& kernel // (n+1) x ci x Ih x Iw
)
{
    const auto& ksize = kernel.size();
    if (ksize != 3 && ksize != 2) throw "fhe::naive::poly: Kernel must be at least a 2 or 3-element vector";
    // if (ksize == 3) return (input * input) * kernel[2] + input * kernel[1] + kernel[0];
    if (ksize == 3)
    {
        auto w1 = kernel[1];
        for (size_t i = 0; i < w1.size(); i++)
            for (size_t h = 0; h < w1[i].size(); h++)
                for (size_t w = 0; w < w1[i][h].size(); w++)
                    w1[i][h][w] /= kernel[2][i][h][w];
        auto r = input * input; // x2
        auto w1x1 = input * w1; // (w1/w2) * x
        r += w1x1; // x2 + (w1/w2) * x
        Ciphertext::refit_inplace(r);
        r *= kernel[2]; // w2 * x2 + w1 * x
        Ciphertext::refit_inplace(r);
        r += kernel[0]; // w2 * x2 + w1 * x + w0
        return r;
    }
    // return (input * input) + input * kernel[1] + kernel[0];
    auto r = input * input;
    r += input * kernel[1];
    Ciphertext::refit_inplace(r);
    r += kernel[0];
    return r;
}

Tensor<3,Ciphertext> // Ci x Ih x Iw
polyskip
(
    const Tensor<3,Ciphertext>& input1, // Ci x Ih x Iw
    const Tensor<3,Ciphertext>& input2, // Ci x Ih x Iw
    const Tensor<4,Scalar>& kernel // 2n x Ci x Ih x Iw
)
{
    const auto& ksize = kernel.size();
    if (ksize != 6 && ksize != 5) throw "fhe::naive::polyskip: Kernel must be at least a 5 or 6-element vector";
    const auto& xa = input1;
    const auto& xb = input2;
    const auto& d0 = kernel[0];
    const auto& db = kernel[1];
    const auto& da = kernel[2];
    const auto& dab= kernel[3];
    const auto& db2= kernel[4];
    
    // [da2] * xa2
    auto xa2 = xa * xa;
    Ciphertext::refit_inplace(xa2);
    if (ksize == 6)
    {
        const auto& da2= kernel[5];
        xa2 *= da2;
        Ciphertext::refit_inplace(xa2);
    }

    // db2 * xb2
    auto xb2 = xb * xb;
    Ciphertext::refit_inplace(xb2);
    xb2 *= db2;
    Ciphertext::refit_inplace(xb2);

    // xa * (xb * dab)
    auto xaxb = xb * dab;
    Ciphertext::refit_inplace(xaxb);
    xaxb *= xa;
    Ciphertext::refit_inplace(xaxb);

    // xa * da
    auto daxa = xa * da;
    Ciphertext::refit_inplace(daxa);

    // xb * db
    auto dbxb = xb * db;
    Ciphertext::refit_inplace(dbxb);

    // output
    auto& r = dbxb;
    r += daxa;
    r += xb2;
    r += xaxb;
    r += xa2;
    r += d0;
    return r;
}

} // naive

} // fhe