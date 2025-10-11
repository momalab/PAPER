#include "embed.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>
#include "adapt.h"
#include "ciphertext.h"
#include "hw.h"
#include "math.h"
#include "plaintext.h"
#include "tensor.h"

using namespace std;
using namespace type;

namespace fhe
{

namespace hw
{

Tensor<2,Ciphertext>
avgpool2d_reformat
(
    const Tensor<2,Ciphertext>& input, // Ci x nCT (Ih x Iw)
    const Configuration& config
)
{
    const auto& kshape = config.kshape();
    const auto& oshape = config.oshape();
    const auto& padding = config.padding();
    const auto& mapping = config.mapping();

    const auto& ci = kshape[1];
    const auto& kh = kshape[2];
    const auto& kw = kshape[3];
    const auto& ph = padding[0];
    const auto& pw = padding[1];
    const auto& oh = oshape[1];
    const auto& ow = oshape[2];
    const auto& slots = Plaintext::default_slots();
    const size_t nCTo = (oh * ow) / slots + ((oh * ow) % slots != 0);
    
    if (ph || pw)
    {
        auto r = avgpool2d(input, config);
        return reformat(r, config);
    }

    const Scalar multiplier = 1.0 / config.divisor();
    Tensor<2,Ciphertext> output{ci, nCTo};
    for (size_t i = 0; i < ci; i++)
    {
        auto pattern = pattern_output(config);
        int64_t prev_shift = 0;
        vector<vector<Ciphertext>> v(nCTo);
        for (size_t krow = 0; krow < kh; krow++)
        {
            for (size_t kcol = 0; kcol < kw; kcol++)
            {
                auto vct = input[i].vector();
                const auto& shift = mapping[krow][kcol];
                int64_t shift_diff = static_cast<int64_t>(shift) - prev_shift;
                prev_shift = shift;
                std::rotate(pattern.rbegin(), pattern.rbegin() + shift_diff, pattern.rend());
                Ciphertext::shiftleft_reformat_inplace(vct, shift, pattern, multiplier);
                for (size_t oct = 0; oct < nCTo; oct++)
                    v[oct].emplace_back(move(vct[oct]));
            }
        }
        for (size_t ict = 0; ict < nCTo; ict++)
            output[i][ict] = move(Ciphertext::add_inplace(v[ict]));
    }
    return output;
}

Tensor<2,Ciphertext> // Co x nCT (Oh x Ow in Ih x Iw)
conv2d_reformat
(
    const Tensor<2,Ciphertext>& input, // Ci x nCT (Ih x Iw)
    const Tensor<5,Plaintext>& kernel, // Co x Ci x Kh x Kw x nCT
    const Tensor<2,Plaintext>& bias, // Co x nCT (Oh x Ow in Ih x Iw)
    const Configuration& config
)
{
    if (input.front_empty()) throw invalid_argument("input");
    if (kernel.front_empty()) throw invalid_argument("kernel");
    if (bias.front_empty()) throw invalid_argument("bias");
    if (input.size() != kernel[0].size()) throw "fhe::hw::conv2d: Input channels in input and kernel do not match";
    if (kernel.size() != bias.size()) throw "fhe::hw::conv2d: Output channels in kernel and bias do not match";

    const auto& kshape = config.kshape();
    const auto& oshape = config.oshape();
    const auto& padding = config.padding();
    const auto& mapping = config.mapping();

    const auto& ci = kshape[1];
    const auto& kh = kshape[2];
    const auto& kw = kshape[3];
    const auto& ph = padding[0];
    const auto& pw = padding[1];
    const auto& co = oshape[0];
    const auto& oh = oshape[1];
    const auto& ow = oshape[2];
    const auto& slots = Plaintext::default_slots();
    const size_t nCTo = (oh * ow) / slots + ((oh * ow) % slots != 0);

    if (ph || pw)
    {
        auto b = encode_bias(Tensor<3,Scalar>{co, oh, ow}, config);
        auto r = conv2d(input, kernel, b, config);
        return reformat(r, config) + bias;
    }

    Tensor<2,Ciphertext> output{co, nCTo};
    for (size_t o = 0; o < co; o++)
    {
        vector<vector<Ciphertext>> v(nCTo);
        for (size_t i = 0; i < ci; i++)
        {
            auto pattern = pattern_output(config);
            int64_t prev_shift = 0;
            for (size_t krow = 0; krow < kh; krow++)
            {
                for (size_t kcol = 0; kcol < kw; kcol++)
                {
                    auto vct = move((input[i] * kernel[o][i][krow][kcol]).vector());
                    const auto& shift = mapping[krow][kcol];
                    int64_t shift_diff = static_cast<int64_t>(shift) - prev_shift;
                    prev_shift = shift;
                    std::rotate(pattern.rbegin(), pattern.rbegin() + shift_diff, pattern.rend());
                    Ciphertext::shiftleft_reformat_inplace(vct, shift, pattern);
                    for (size_t oct = 0; oct < nCTo; oct++)
                        v[oct].emplace_back(move(vct[oct]));
                }
            }
        }
        for (size_t oct = 0; oct < nCTo; oct++)
            output[o][oct] = Ciphertext::add_inplace(v[oct]) + bias[o][oct];
    }
    return output;
}

Tensor<2,Ciphertext> // 1 x nCTo (1 x Ko) // each ko uses 1 slot
linear_reformat
(
    const Tensor<2,Ciphertext>& input, // Ci x nCT (Ih x Iw)
    const Tensor<3,Plaintext>& kernel, // Ko x Ci x nCT (Ih x Iw)
    const Tensor<2,Plaintext>& bias, // 1 x nCTo (1 x Ko)
    const Configuration& config
)
{
    const auto& ishape = config.ishape();
    const auto& kshape = config.kshape();
    const auto& mapping = config.mapping();

    const auto& ci = ishape[0];
    const auto& ko = kshape[0];
    const auto& slots = Plaintext::default_slots();
    const size_t nCTo = ko / slots + (ko % slots != 0);
    size_t max_slot = min(1UL << util::ceil_log2(mapping.back().back() + 1), slots);

    Plaintext mask;
    {
        vector<Scalar> v(slots, 0);
        v[0] = 1;
        mask = move(Plaintext(v));
    }

    Tensor<2,Ciphertext> output{1, nCTo};
    vector<vector<Ciphertext>> v(nCTo);
    for (size_t o = 0; o < ko; o++) // compute each output value
    {
        for (size_t i = 0; i < ci; i++)
        {
            // apply transformation and sum if more than one ciphertext
            auto sum = move(Ciphertext::add_inplace((input[i] * kernel[o][i]).vector()));

            // shift and add intermediate values to get one output value
            Ciphertext::addslots_inplace(sum, max_slot);
            sum *= mask;

            // position output value in the right slot
            size_t idx = o;
            size_t idxdiv = idx / slots;
            size_t idxmod = idx % slots;
            sum >>= idxmod;
            v[idxdiv].emplace_back(move(sum));
        }
    }
    for (size_t ict = 0; ict < nCTo; ict++)
        output[0][ict] = Ciphertext::add_inplace(v[ict]) + bias[0][ict];
    return output;
}

} // hw

} // fhe