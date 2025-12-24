#include "conv.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>
#include "ciphertext.h"
#include "defines.h"
#include "encode.h"
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

Tensor<2,Ciphertext> // Co x nCT (Oh x Ow in Ih x Iw)
conv2d
(
    Tensor<2,Ciphertext>& input, // Ci x nCT (Ih x Iw)
    const Tensor<6,Scalar>& kernel, // scalar tensor
    const Tensor<3,Scalar>& bias, // Co x nCT x Ih x Iw (Oh x Ow in Ih x Iw)
    const Configuration& config,
    size_t nthreads
)
{
    if (input.front_empty()) throw invalid_argument("input");
    if (kernel.front_empty()) throw invalid_argument("kernel");
    if (bias.front_empty()) throw invalid_argument("bias");
    if (input.size() != kernel[0][0][0].size()) throw "fhe::hw::conv2d: Input channels in input and kernel do not match";
    if (kernel.size() != bias.size()) throw "fhe::hw::conv2d: Output channels in kernel and bias do not match";
    
    const auto& kshape = config.kshape();

    const auto& co = kshape[0];
    const auto& ci = kshape[1];
    const auto& kh = kshape[2];
    const auto& kw = kshape[3];
    const auto& nCT = config.nct();
    const size_t ksize = kh * kw;
    
    size_t nthreads_ci = min(nthreads, ci);
    size_t nthreads_hwi= max(min(nthreads / nthreads_ci, ksize), 1UL);
    size_t nthreads_co = min(nthreads, co);
    size_t nthreads_hw = max(min(nthreads / nthreads_co, ksize), 1UL);
    
    mutex mtx;
    exception_ptr exception;

    Tensor<3,Ciphertext> shifted_input{ksize, ci, nCT}; // shifted input
    vector<thread> threads_ci(nthreads_ci);
    for (size_t thr_ci = 0; thr_ci < nthreads_ci; thr_ci++) threads_ci[thr_ci] = thread([&, thr_ci]()
    {
        try
        {
            for (size_t i = thr_ci; i < ci; i += nthreads_ci)
            {
                Ciphertext::regularize_inplace(input[i]); // regularize input
                vector<thread> threads_hw(nthreads_hwi);
                for (size_t thr_hw = 0; thr_hw < nthreads_hwi; thr_hw++) threads_hw[thr_hw] = thread([&, thr_hw]()
                {
                    try
                    {
                        for (size_t k = thr_hw; k < ksize; k += nthreads_hwi)
                        {
                            size_t krow = k / kw, kcol = k % kw;
                            int shift = config.shift(krow, kcol);
                            shifted_input[k][i] = input[i];
                            Ciphertext::shiftleft_inplace(shifted_input[k][i].vector(), shift);
                        }
                    }
                    catch (...)
                    {
                        lock_guard<mutex> lock(mtx);
                        if (!exception) exception = current_exception();
                    }
                });
                for (auto& thread : threads_hw) thread.join();
                if (exception) rethrow_exception(exception);
            }
        }
        catch (...)
        {
            lock_guard<mutex> lock(mtx);
            if (!exception) exception = current_exception();
        }
    });
    for (auto& thread : threads_ci) thread.join();
    if (exception) rethrow_exception(exception);

    size_t nthreads_ew = max(nthreads / (nthreads_co * nthreads_hw), 1UL);
    size_t nthreads_eb = max(nthreads / nthreads_co, 1UL);
    int level = shifted_input[0][0][0].level();
    double keyscale = shifted_input[0][0][0].keyscale();

    Tensor<2,Ciphertext> output{co, nCT};
    vector<thread> threads_co(nthreads_co);
    for (size_t thr_co = 0; thr_co < nthreads_co; thr_co++) threads_co[thr_co] = thread([&, thr_co]()
    {
        try
        {
            for (size_t o = thr_co; o < co; o += nthreads_co)
            {
                Tensor<2,Ciphertext> v{ksize, nCT};
                vector<thread> threads_hw(nthreads_hw);
                for (size_t thr_hw = 0; thr_hw < nthreads_hw; thr_hw++) threads_hw[thr_hw] = thread([&, thr_hw]()
                {
                    try
                    {
                        for (size_t k = thr_hw; k < ksize; k += nthreads_hw)
                        {
                            size_t krow = k / kw, kcol = k % kw;
                            int shift = config.shift(krow, kcol);
                            for (size_t i = 0; i < ci; i++)
                            {
                                auto weight = encode_weight(kernel[o][krow][kcol][i], krow, kcol, shift, config, nthreads_ew, level, keyscale);
                                for (size_t ict = 0; ict < nCT; ict++)
                                {
                                    auto product = shifted_input[k][i][ict] * weight[ict]; // apply filter
                                    if (!i) v[k][ict] = move(product);
                                    else v[k][ict] += product;
                                }
                            }
                        }
                    }
                    catch (...)
                    {
                        lock_guard<mutex> lock(mtx);
                        if (!exception) exception = current_exception();
                    }
                });
                for (auto& thread : threads_hw) thread.join();
                if (exception) rethrow_exception(exception);
                output[o] = move(v[0]);
                for (size_t k = 1; k < ksize; k++) output[o] += v[k];
                output[o] += encode_bias(bias[o], config, nthreads_eb, output[o][0].level(), output[o][0].scale());
            }
        }
        catch (...)
        {
            lock_guard<mutex> lock(mtx);
            if (!exception) exception = current_exception();
        }
    });
    for (auto& thread : threads_co) thread.join();
    if (exception) rethrow_exception(exception);
    
    return output;
}

Tensor<2,Ciphertext> // Co x nCT (Oh x Ow in Ih x Iw)
conv2d
(
    Tensor<2,Ciphertext>& input, // Ci x nCT (Ih x Iw)
    const Tensor<7,Scalar>& kernel, // scalar tensor
    const Tensor<4,Scalar>& bias, // Co x nCT x Ih x Iw (Oh x Ow in Ih x Iw)
    const Configuration& config,
    size_t nthreads,
    const vector<int>& offsets
)
{
    if (input.front_empty()) throw invalid_argument("input");
    if (kernel.front_empty()) throw invalid_argument("kernel");
    if (bias.front_empty()) throw invalid_argument("bias");
    if (kernel.size() != offsets.size()) throw "fhe::hw::conv2d: Number of models in kernel and offsets do not match";
    if (bias.size() != offsets.size()) throw "fhe::hw::conv2d: Number of models in bias and offsets do not match";
    if (input.size() != kernel[0][0][0][0].size()) throw "fhe::hw::conv2d: Input channels in input and kernel do not match";
    if (kernel[0].size() != bias[0].size()) throw "fhe::hw::conv2d: Output channels in kernel and bias do not match";

    const auto& kshape = config.kshape();

    const auto& co = kshape[0];
    const auto& ci = kshape[1];
    const auto& kh = kshape[2];
    const auto& kw = kshape[3];
    const auto& nCT = config.nct();
    const size_t ksize = kh * kw;
    size_t nModels = kernel.size();
    
    size_t nthreads_ci = min(nthreads, ci);
    size_t nthreads_hwi= max(min(nthreads / nthreads_ci, ksize), 1UL);
    size_t nthreads_co = min(nthreads, co);
    size_t nthreads_hw = max(min(nthreads / nthreads_co, ksize), 1UL);
    
    mutex mtx;
    exception_ptr exception;

    Tensor<3,Ciphertext> shifted_input{ksize, ci, nCT}; // shifted input
    vector<thread> threads_ci(nthreads_ci);
    for (size_t thr_ci = 0; thr_ci < nthreads_ci; thr_ci++) threads_ci[thr_ci] = thread([&, thr_ci]()
    {
        try
        {
            for (size_t i = thr_ci; i < ci; i += nthreads_ci)
            {
                Ciphertext::regularize_inplace(input[i]); // regularize input
                vector<thread> threads_hw(nthreads_hwi);
                for (size_t thr_hw = 0; thr_hw < nthreads_hwi; thr_hw++) threads_hw[thr_hw] = thread([&, thr_hw]()
                {
                    try
                    {
                        for (size_t k = thr_hw; k < ksize; k += nthreads_hwi)
                        {
                            size_t krow = k / kw, kcol = k % kw;
                            int shift = config.shift(krow, kcol);
                            shifted_input[k][i] = input[i];
                            Ciphertext::shiftleft_inplace(shifted_input[k][i].vector(), shift);
                        }
                    }
                    catch (...)
                    {
                        lock_guard<mutex> lock(mtx);
                        if (!exception) exception = current_exception();
                    }
                });
                for (auto& thread : threads_hw) thread.join();
                if (exception) rethrow_exception(exception);
            }
        }
        catch (...)
        {
            lock_guard<mutex> lock(mtx);
            if (!exception) exception = current_exception();
        }
    });
    for (auto& thread : threads_ci) thread.join();
    if (exception) rethrow_exception(exception);

    size_t nthreads_ew = max(nthreads / (nthreads_co * nthreads_hw), 1UL);
    size_t nthreads_eb = max(nthreads / nthreads_co, 1UL);
    int level = shifted_input[0][0][0].level();
    double keyscale = shifted_input[0][0][0].keyscale();

    Tensor<2,Ciphertext> output{co, nCT};
    vector<thread> threads_co(nthreads_co);
    for (size_t thr_co = 0; thr_co < nthreads_co; thr_co++) threads_co[thr_co] = thread([&, thr_co]()
    {
        try
        {
            for (size_t o = thr_co; o < co; o += nthreads_co)
            {
                Tensor<2,Ciphertext> v{ksize, nCT};
                vector<thread> threads_hw(nthreads_hw);
                for (size_t thr_hw = 0; thr_hw < nthreads_hw; thr_hw++) threads_hw[thr_hw] = thread([&, thr_hw]()
                {
                    try
                    {
                        for (size_t k = thr_hw; k < ksize; k += nthreads_hw)
                        {
                            size_t krow = k / kw, kcol = k % kw;
                            int shift = config.shift(krow, kcol);
                            for (size_t i = 0; i < ci; i++)
                            {
                                vector<Scalar> vs;
                                for (size_t m = 0; m < nModels; m++) vs.push_back(kernel[m][o][krow][kcol][i][0][0]);
                                auto weight = encode_weight(vs, krow, kcol, shift, config, nthreads_ew, level, keyscale, offsets);
                                for (size_t ict = 0; ict < nCT; ict++)
                                {
                                    auto product = shifted_input[k][i][ict] * weight[ict]; // apply filter
                                    if (!i) v[k][ict] = move(product);
                                    else v[k][ict] += product;
                                }
                            }
                        }
                    }
                    catch (...)
                    {
                        lock_guard<mutex> lock(mtx);
                        if (!exception) exception = current_exception();
                    }
                });
                for (auto& thread : threads_hw) thread.join();
                if (exception) rethrow_exception(exception);
                output[o] = move(v[0]);
                for (size_t k = 1; k < ksize; k++) output[o] += v[k];
                output[o] += encode_bias(bias, o, config, nthreads_eb, output[o][0].level(), output[o][0].scale(), offsets);
            }
        }
        catch (...)
        {
            lock_guard<mutex> lock(mtx);
            if (!exception) exception = current_exception();
        }
    });
    for (auto& thread : threads_co) thread.join();
    if (exception) rethrow_exception(exception);

    return output;
}

} // hw

} // fhe