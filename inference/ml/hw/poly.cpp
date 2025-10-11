#include "poly.h"

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
poly
(
    Tensor<2,Ciphertext>& input, // Ci x nCT (Ih x Iw)
    const Tensor<4,Scalar>& kernel, // (n+1) x Ci x Ih x Iw
    const Configuration& config,
    size_t nthreads
)
{
    const auto& ksize = kernel.size();
    if (ksize != 3 && ksize != 2) throw invalid_argument("fhe::hw::poly: Kernel must be a 3-element vector");

    auto& x = input;

    const auto& ci = x.size();
    const auto& nct = x[0].size();
    size_t nthreads_ci = min(nthreads, ci);
    size_t nthreads_ek = max(nthreads / nthreads_ci, 1UL);
    mutex mtx;
    exception_ptr exception;
    Tensor<2,Ciphertext> r{ci, nct};
    vector<thread> threads_ci(nthreads_ci);
    for (size_t thr_ci = 0; thr_ci < nthreads_ci; thr_ci++) threads_ci[thr_ci] = thread([&, thr_ci]()
    {
        try
        {
            for (size_t i = thr_ci; i < ci; i += nthreads_ci)
            {
                Ciphertext::refit_inplace(x[i]);
                int level = x[i][0].level();
                double scale = x[i][0].scale();

                auto w2x2 = x[i] * x[i]; // x2
                Ciphertext::relinearize_inplace(w2x2);
                if (ksize == 3)
                {
                    w2x2 *= encode_kernel(kernel, 2UL, i, config, nthreads_ek, &x[i][0]); // w2 * x2
                    scale *= scale;
                }
                r[i] *= encode_kernel(kernel, 1UL, i, config, nthreads_ek, level, scale); // w1 * x1
                r[i] += w2x2;
                r[i] += encode_kernel(kernel, 0UL, i, config, nthreads_ek, &r[i][0]);
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
    return r;
}

Tensor<2,Ciphertext> // Ci x nCT (Ih x Iw)
poly
(
    Tensor<2,Ciphertext>& input, // Ci x nCT (Ih x Iw)
    const Tensor<5,Scalar>& kernel, // (# models) x (n+1) x Ci x Ih x Iw
    const Configuration& config,
    size_t nthreads,
    const vector<int>& offsets
)
{
    if (kernel.empty()) throw invalid_argument("fhe::hw::poly_inplace: Kernel is empty");
    if (offsets.size() != kernel.size()) throw invalid_argument("fhe::hw::poly_inplace: Offsets size must match kernel size");
    const auto& ksize = kernel[0].size();
    if (ksize != 3 && ksize != 2) throw invalid_argument("fhe::hw::poly: Kernel must be a 3-element vector");

    auto& x = input;

    const auto& ci = x.size();
    const auto& nct = x[0].size();
    size_t nthreads_ci = min(nthreads, ci);
    size_t nthreads_ek = max(nthreads / nthreads_ci, 1UL);
    mutex mtx;
    exception_ptr exception;
    Tensor<2,Ciphertext> r{ci, nct};
    vector<thread> threads_ci(nthreads_ci);
    for (size_t thr_ci = 0; thr_ci < nthreads_ci; thr_ci++) threads_ci[thr_ci] = thread([&, thr_ci]()
    {
        try
        {
            for (size_t i = thr_ci; i < ci; i += nthreads_ci)
            {
                Ciphertext::refit_inplace(x[i]);
                int level = x[i][0].level();
                double scale = x[i][0].scale();

                auto w2x2 = x[i] * x[i]; // x2
                Ciphertext::relinearize_inplace(w2x2);
                if (ksize == 3)
                {
                    w2x2 *= encode_kernel(kernel, 2UL, i, config, nthreads_ek, &x[i][0], offsets); // w2 * x2
                    scale *= scale;
                }
                r[i] *= encode_kernel(kernel, 1UL, i, config, nthreads_ek, level, scale, offsets); // w1 * x1
                r[i] += w2x2;
                r[i] += encode_kernel(kernel, 0UL, i, config, nthreads_ek, &r[i][0], offsets);
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
    return r;
}

void // Ci x nCT (Ih x Iw)
poly_inplace
(
    Tensor<2,Ciphertext>& input, // Ci x nCT (Ih x Iw)
    const Tensor<4,Scalar>& kernel, // (n+1) x Ci x Ih x Iw
    const Configuration& config,
    size_t nthreads
)
{
    const auto& ksize = kernel.size();
    if (ksize != 3 && ksize != 2) throw invalid_argument("fhe::hw::poly: Kernel must be a 3-element vector");

    auto& x = input;

    const auto& ci = x.size();
    size_t nthreads_ci = min(nthreads, ci);
    size_t nthreads_ek = max(nthreads / nthreads_ci, 1UL);
    mutex mtx;
    exception_ptr exception;
    vector<thread> threads_ci(nthreads_ci);
    for (size_t thr_ci = 0; thr_ci < nthreads_ci; thr_ci++) threads_ci[thr_ci] = thread([&, thr_ci]()
    {
        try
        {
            for (size_t i = thr_ci; i < ci; i += nthreads_ci)
            {
                Ciphertext::refit_inplace(x[i]);
                int level = x[i][0].level();
                double scale = x[i][0].scale();

                auto w2x2 = x[i] * x[i]; // x2
                Ciphertext::relinearize_inplace(w2x2);
                if (ksize == 3)
                {
                    w2x2 *= encode_kernel(kernel, 2UL, i, config, nthreads_ek, &x[i][0]); // w2 * x2
                    scale *= scale;
                }
                x[i] *= encode_kernel(kernel, 1UL, i, config, nthreads_ek, level, scale); // w1 * x1
                x[i] += w2x2;
                x[i] += encode_kernel(kernel, 0UL, i, config, nthreads_ek, &x[i][0]);
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

void // Ci x nCT (Ih x Iw)
poly_inplace
(
    Tensor<2,Ciphertext>& input, // Ci x nCT (Ih x Iw)
    const Tensor<5,Scalar>& kernel, // (# models) x (n+1) x Ci x Ih x Iw
    const Configuration& config,
    size_t nthreads,
    const vector<int>& offsets
)
{
    if (kernel.empty()) throw invalid_argument("fhe::hw::poly_inplace: Kernel is empty");
    if (offsets.size() != kernel.size()) throw invalid_argument("fhe::hw::poly_inplace: Offsets size must match kernel size");
    const auto& ksize = kernel[0].size();
    if (ksize != 3 && ksize != 2) throw invalid_argument("fhe::hw::poly: Kernel must be a 3-element vector");

    auto& x = input;

    const auto& ci = x.size();
    size_t nthreads_ci = min(nthreads, ci);
    size_t nthreads_ek = max(nthreads / nthreads_ci, 1UL);
    mutex mtx;
    exception_ptr exception;
    vector<thread> threads_ci(nthreads_ci);
    for (size_t thr_ci = 0; thr_ci < nthreads_ci; thr_ci++) threads_ci[thr_ci] = thread([&, thr_ci]()
    {
        try
        {
            for (size_t i = thr_ci; i < ci; i += nthreads_ci)
            {
                Ciphertext::refit_inplace(x[i]);
                int level = x[i][0].level();
                double scale = x[i][0].scale();

                auto w2x2 = x[i] * x[i]; // x2
                Ciphertext::relinearize_inplace(w2x2);
                if (ksize == 3)
                {
                    w2x2 *= encode_kernel(kernel, 2UL, i, config, nthreads_ek, &x[i][0], offsets); // w2 * x2
                    scale *= scale;
                }
                x[i] *= encode_kernel(kernel, 1UL, i, config, nthreads_ek, level, scale, offsets); // w1 * x1
                x[i] += w2x2;
                x[i] += encode_kernel(kernel, 0UL, i, config, nthreads_ek, &x[i][0], offsets);
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