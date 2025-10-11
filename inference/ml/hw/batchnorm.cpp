#include "batchnorm.h"

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

Tensor<2,Ciphertext> // Ci x nCT (Ih x Iw)
batchnorm2d
(
    Tensor<2,Ciphertext>& input, // Ci x nCT (Ih x Iw)
    const Tensor<4,Scalar>& kernel, // (n+1) x Ci x nCT Ih x Iw
    const Configuration& config,
    size_t nthreads
)
{
    const auto& ksize = kernel.size();
    if (ksize != 2 && ksize != 1) throw invalid_argument("fhe::hw::batchnorm: Kernel must be a 1 or 2-element vector");

    const auto& ci = input.size();
    size_t nthreads_ci = min(nthreads, ci);
    size_t nthreads_enc = max(nthreads / nthreads_ci, 1UL);
    
    mutex mtx;
    exception_ptr exception;
    vector<thread> threads_ci(nthreads_ci);
    Tensor<2,Ciphertext> output(input.shape());
    if (ksize == 2) for (size_t thr_ci = 0; thr_ci < nthreads_ci; thr_ci++) threads_ci[thr_ci] = thread([&, thr_ci]()
    {
        try
        {
            for (size_t i = thr_ci; i < ci; i += nthreads_ci)
            {
                Ciphertext::regularize_inplace(input[i]);
                output[i] = input[i] * encode_kernel(kernel, 1UL, i, config, nthreads_enc, &input[i][0]);
                output[i] += encode_kernel(kernel, 0UL, i, config, nthreads_enc, &output[i][0]);
            }
        }
        catch (...)
        {
            lock_guard<mutex> lock(mtx);
            if (!exception) exception = current_exception();
        }
    });
    else for (size_t thr_ci = 0; thr_ci < nthreads_ci; thr_ci++) threads_ci[thr_ci] = thread([&, thr_ci]()
    {
        try
        {
            for (size_t i = thr_ci; i < ci; i += nthreads_ci)
            {
                Ciphertext::regularize_inplace(input[i]);
                output[i] = input[i] + encode_kernel(kernel, 0UL, i, config, nthreads_enc, &input[i][0]);
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

    return output;
}

Tensor<2,Ciphertext> // Ci x nCT (Ih x Iw)
batchnorm2d
(
    Tensor<2,Ciphertext>& input, // Ci x nCT (Ih x Iw)
    const Tensor<5,Scalar>& kernel, // (# models) x (n+1) x Ci x nCT Ih x Iw
    const Configuration& config,
    size_t nthreads,
    const vector<int>& offsets
)
{
    if (kernel.empty()) throw invalid_argument("fhe::hw::batchnorm_inplace: Kernel is empty");
    if (kernel.size() != offsets.size()) throw invalid_argument("fhe::hw::batchnorm_inplace: Kernel and offsets size mismatch");
    const auto& ksize = kernel[0].size();
    if (ksize != 2 && ksize != 1) throw invalid_argument("fhe::hw::batchnorm: Kernel must be a 1 or 2-element vector");

    const auto& ci = input.size();
    size_t nthreads_ci = min(nthreads, ci);
    size_t nthreads_enc = max(nthreads / nthreads_ci, 1UL);
    
    mutex mtx;
    exception_ptr exception;
    vector<thread> threads_ci(nthreads_ci);
    Tensor<2,Ciphertext> output(input.shape());
    if (ksize == 2) for (size_t thr_ci = 0; thr_ci < nthreads_ci; thr_ci++) threads_ci[thr_ci] = thread([&, thr_ci]()
    {
        try
        {
            for (size_t i = thr_ci; i < ci; i += nthreads_ci)
            {
                Ciphertext::regularize_inplace(input[i]);
                output[i] = input[i] * encode_kernel(kernel, 1UL, i, config, nthreads_enc, &input[i][0], offsets);
                output[i] += encode_kernel(kernel, 0UL, i, config, nthreads_enc, &output[i][0], offsets);
            }
        }
        catch (...)
        {
            lock_guard<mutex> lock(mtx);
            if (!exception) exception = current_exception();
        }
    });
    else for (size_t thr_ci = 0; thr_ci < nthreads_ci; thr_ci++) threads_ci[thr_ci] = thread([&, thr_ci]()
    {
        try
        {
            for (size_t i = thr_ci; i < ci; i += nthreads_ci)
            {
                Ciphertext::regularize_inplace(input[i]);
                output[i] = input[i] + encode_kernel(kernel, 0UL, i, config, nthreads_enc, &input[i][0], offsets);
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

    return output;
}

void batchnorm2d_inplace
(
    Tensor<2,Ciphertext>& input, // Ci x nCT (Ih x Iw)
    const Tensor<4,Scalar>& kernel, // (n+1) x Ci x nCT Ih x Iw
    const Configuration& config,
    size_t nthreads
)
{
    const auto& ksize = kernel.size();
    if (ksize != 2 && ksize != 1) throw invalid_argument("fhe::hw::batchnorm_inplace: Kernel must be a 1 or 2-element vector");

    const auto& ci = input.size();
    size_t nthreads_ci = min(nthreads, ci);
    size_t nthreads_enc = max(nthreads / nthreads_ci, 1UL);
    
    mutex mtx;
    exception_ptr exception;
    vector<thread> threads_ci(nthreads_ci);
    if (ksize == 2) for (size_t thr_ci = 0; thr_ci < nthreads_ci; thr_ci++) threads_ci[thr_ci] = thread([&, thr_ci]()
    {
        try
        {
            for (size_t i = thr_ci; i < ci; i += nthreads_ci)
            {
                Ciphertext::regularize_inplace(input[i]);
                input[i] *= encode_kernel(kernel, 1UL, i, config, nthreads_enc, input[i][0].level(), input[i][0].keyscale());
                input[i] += encode_kernel(kernel, 0UL, i, config, nthreads_enc, input[i][0].level(), input[i][0].scale());
            }
        }
        catch (...)
        {
            lock_guard<mutex> lock(mtx);
            if (!exception) exception = current_exception();
        }
    });
    else for (size_t thr_ci = 0; thr_ci < nthreads_ci; thr_ci++) threads_ci[thr_ci] = thread([&, thr_ci]()
    {
        try
        {
            for (size_t i = thr_ci; i < ci; i += nthreads_ci)
                input[i] += encode_kernel(kernel, 0UL, i, config, nthreads_enc, &input[i][0]);
        }
        catch (...)
        {
            lock_guard<mutex> lock(mtx);
            if (!exception) exception = current_exception();
        }
    });
    for (auto& thread : threads_ci) thread.join();
    if (exception) rethrow_exception(exception);
}

void batchnorm2d_inplace
(
    Tensor<2,Ciphertext>& input, // Ci x nCT (Ih x Iw)
    const Tensor<5,Scalar>& kernel, // (# models) x (n+1) x Ci x nCT Ih x Iw
    const Configuration& config,
    size_t nthreads,
    const vector<int>& offsets
)
{
    if (kernel.empty()) throw invalid_argument("fhe::hw::batchnorm_inplace: Kernel is empty");
    if (kernel.size() != offsets.size()) throw invalid_argument("fhe::hw::batchnorm_inplace: Kernel and offsets size mismatch");
    const auto& ksize = kernel[0].size();
    if (ksize != 2 && ksize != 1) throw invalid_argument("fhe::hw::batchnorm_inplace: Kernel must be a 1 or 2-element vector");

    const auto& ci = input.size();
    size_t nthreads_ci = min(nthreads, ci);
    size_t nthreads_enc = max(nthreads / nthreads_ci, 1UL);
    
    mutex mtx;
    exception_ptr exception;
    vector<thread> threads_ci(nthreads_ci);
    if (ksize == 2) for (size_t thr_ci = 0; thr_ci < nthreads_ci; thr_ci++) threads_ci[thr_ci] = thread([&, thr_ci]()
    {
        try
        {
            for (size_t i = thr_ci; i < ci; i += nthreads_ci)
            {
                Ciphertext::regularize_inplace(input[i]);
                input[i] *= encode_kernel(kernel, 1UL, i, config, nthreads_enc, input[i][0].level(), input[i][0].keyscale(), offsets);
                input[i] += encode_kernel(kernel, 0UL, i, config, nthreads_enc, input[i][0].level(), input[i][0].scale(), offsets);
            }
        }
        catch (...)
        {
            lock_guard<mutex> lock(mtx);
            if (!exception) exception = current_exception();
        }
    });
    else for (size_t thr_ci = 0; thr_ci < nthreads_ci; thr_ci++) threads_ci[thr_ci] = thread([&, thr_ci]()
    {
        try
        {
            for (size_t i = thr_ci; i < ci; i += nthreads_ci)
            {
                input[i] += encode_kernel(kernel, 0UL, i, config, nthreads_enc, &input[i][0], offsets);
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
}

} // hw

} // fhe