#include "linear.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

Tensor<2,Ciphertext> // 1 x nCTo (Ko x Ih x Iw) // each ko uses Ih x Iw slots
linear
(
    Tensor<2,Ciphertext>& input, // Ci x nCT (Ih x Iw)
    const Tensor<4,Scalar>& kernel, // Ko x Ci x Ih x Iw
    const Tensor<3,Scalar>& bias, // 1 x 1 x Ko
    const Configuration& config,
    size_t nthreads
)
{
    const auto& ishape = config.ishape();
    const auto& kshape = config.kshape();
    const auto& imap = config.imap();

    const auto& ci = ishape[0];
    const auto& ih = ishape[1];
    const auto& iw = ishape[2];
    const auto& ko = kshape[0];
    const auto& nCTi = config.ncti();
    const auto& nCTo = config.ncto();
    const size_t hw = ih * iw;

    size_t nthreads_ko = max(min(nthreads, ko), 1UL);
    size_t nthreads_ci = max(min(nthreads / nthreads_ko, ci), 1UL);
    size_t nthreads_hw = max(min(nthreads / nthreads_ko, hw), 1UL);
    size_t nthreads_bias = max(nthreads / nthreads_ko, 1UL);

    mutex mtx;
    exception_ptr exception;
    size_t nthreads_ek = max(nthreads / (nthreads_ko * nthreads_ci), 1UL);
    Tensor<2,Ciphertext> output{ko, nCTo};

    // Linear transformation and addition of input channels
    vector<thread> threads_ko(nthreads_ko);
    for (size_t thr_ko = 0; thr_ko < nthreads_ko; thr_ko++) threads_ko[thr_ko] = thread([&, thr_ko]()
    {
        try
        {
            for (size_t o = thr_ko; o < ko; o += nthreads_ko) // compute each output value
            {
                Tensor<2,Ciphertext> vct{ci, nCTi};
                vector<thread> threads_ci(nthreads_ci);
                for (size_t thr_ci = 0; thr_ci < nthreads_ci; thr_ci++) threads_ci[thr_ci] = thread([&, thr_ci, o]()
                {
                    try
                    {
                        for (size_t i = thr_ci; i < ci; i += nthreads_ci) // apply transformation
                            vct[i] = input[i] * encode_kernel(kernel, o, i, config, nthreads_ek, input[i][0].level(), input[i][0].keyscale());
                    }
                    catch (...)
                    {
                        lock_guard<mutex> lock(mtx);
                        if (!exception) exception = current_exception();
                    }
                });
                for (auto& thread : threads_ci) thread.join();
                if (exception) rethrow_exception(exception);

                // Add all input channels for the current output channel
                for (size_t i = 1; i < ci; i++) vct[0] += vct[i];
                auto& sum = vct[0]; // sum is the accumulated value of all input channels for the current output channel

                // Shift and add
                Tensor<2,Ciphertext> shifted{hw, nCTi};
                vector<thread> threads_hw(nthreads_hw);
                for (size_t thr_hw = 0; thr_hw < nthreads_hw; thr_hw++) threads_hw[thr_hw] = thread([&, thr_hw]()
                {
                    try
                    {
                        for (size_t i = thr_hw; i < hw; i += nthreads_hw)
                        {
                            size_t ii = i / iw, ij = i % iw;
                            size_t idx = imap[ii][ij];
                            shifted[i] = sum; // copy the accumulated value to the shifted tensor
                            Ciphertext::shiftleft_inplace(shifted[i].vector(), idx); // shift the output value to the right position
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

                // Sum all shifted values
                for (size_t i = 1; i < hw; i++) shifted[0][0] += shifted[i][0];

                output[o][0] = shifted[0][0];
                output[o] += encode_bias(bias, o, config, nthreads_bias, output[o][0].level(), output[o][0].scale()); // add bias
            }
        }
        catch (...)
        {
            lock_guard<mutex> lock(mtx);
            if (!exception) exception = current_exception();
        }
    });
    for (auto& thread : threads_ko) thread.join();
    if (exception) rethrow_exception(exception);

    return output;
}

Tensor<2,Ciphertext> // 1 x nCTo (Ko x Ih x Iw) // each ko uses Ih x Iw slots
linear
(
    Tensor<2,Ciphertext>& input, // Ci x nCT (Ih x Iw)
    const Tensor<5,Scalar>& kernel, // Ko x Ci x Ih x Iw
    const Tensor<4,Scalar>& bias, // 1 x 1 x Ko
    const Configuration& config,
    size_t nthreads,
    const vector<int>& offsets
)
{
    const auto& ishape = config.ishape();
    const auto& kshape = config.kshape();
    const auto& imap = config.imap();

    const auto& ci = ishape[0];
    const auto& ih = ishape[1];
    const auto& iw = ishape[2];
    const auto& ko = kshape[0];
    const auto& nCTi = config.ncti();
    const auto& nCTo = config.ncto();
    const size_t hw = ih * iw;

    size_t nthreads_ko = max(min(nthreads, ko), 1UL);
    size_t nthreads_ci = max(min(nthreads / nthreads_ko, ci), 1UL);
    size_t nthreads_hw = max(min(nthreads / nthreads_ko, hw), 1UL);
    size_t nthreads_bias = max(nthreads / nthreads_ko, 1UL);

    mutex mtx;
    exception_ptr exception;
    size_t nthreads_ek = max(nthreads / (nthreads_ko * nthreads_ci), 1UL);
    Tensor<2,Ciphertext> output{ko, nCTo};

    // Linear transformation and addition of input channels
    vector<thread> threads_ko(nthreads_ko);
    for (size_t thr_ko = 0; thr_ko < nthreads_ko; thr_ko++) threads_ko[thr_ko] = thread([&, thr_ko]()
    {
        try
        {
            for (size_t o = thr_ko; o < ko; o += nthreads_ko) // compute each output value
            {
                Tensor<2,Ciphertext> vct{ci, nCTi};
                vector<thread> threads_ci(nthreads_ci);
                for (size_t thr_ci = 0; thr_ci < nthreads_ci; thr_ci++) threads_ci[thr_ci] = thread([&, thr_ci, o]()
                {
                    try
                    {
                        for (size_t i = thr_ci; i < ci; i += nthreads_ci) // apply transformation
                            vct[i] = input[i] * encode_kernel(kernel, o, i, config, nthreads_ek, input[i][0].level(), input[i][0].keyscale(), offsets);
                    }
                    catch (...)
                    {
                        lock_guard<mutex> lock(mtx);
                        if (!exception) exception = current_exception();
                    }
                });
                for (auto& thread : threads_ci) thread.join();
                if (exception) rethrow_exception(exception);

                // Add all input channels for the current output channel
                for (size_t i = 1; i < ci; i++) vct[0] += vct[i];
                auto& sum = vct[0]; // sum is the accumulated value of all input channels for the current output channel

                // Shift and add
                Tensor<2,Ciphertext> shifted{hw, nCTi};
                vector<thread> threads_hw(nthreads_hw);
                for (size_t thr_hw = 0; thr_hw < nthreads_hw; thr_hw++) threads_hw[thr_hw] = thread([&, thr_hw]()
                {
                    try
                    {
                        for (size_t i = thr_hw; i < hw; i += nthreads_hw)
                        {
                            size_t ii = i / iw, ij = i % iw;
                            size_t idx = imap[ii][ij];
                            shifted[i] = sum; // copy the accumulated value to the shifted tensor
                            Ciphertext::shiftleft_inplace(shifted[i].vector(), idx); // shift the output value to the right position
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

                // Sum all shifted values
                for (size_t i = 1; i < hw; i++) shifted[0][0] += shifted[i][0];

                output[o][0] = shifted[0][0];
                output[o] += encode_bias(bias, o, config, nthreads_bias, output[o][0].level(), output[o][0].scale(), offsets); // add bias
            }
        }
        catch (...)
        {
            lock_guard<mutex> lock(mtx);
            if (!exception) exception = current_exception();
        }
    });
    for (auto& thread : threads_ko) thread.join();
    if (exception) rethrow_exception(exception);

    return output;
}

} // hw

} // fhe