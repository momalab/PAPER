#include "avgpool.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>
#include "auxiliary.h"
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

Tensor<2,Ciphertext>
avgpool2d
(
    Tensor<2,Ciphertext>& input, // Ci x nCT (Ih x Iw)
    const Configuration& config,
    size_t nthreads
)
{
    const auto& kshape = config.kshape();

    const auto& ci = kshape[1];
    const auto& kh = kshape[2];
    const auto& kw = kshape[3];
    const size_t ksize = kh * kw;
    Scalar multiplier = 1.0 / config.divisor();
    size_t nthreads_ci = max(min(nthreads, ci), 1UL);

    mutex mtx;
    exception_ptr exception;
    vector<thread> threads_ci(nthreads_ci);
    // regularize input
    for (size_t thr_ci = 0; thr_ci < nthreads_ci; thr_ci++) threads_ci[thr_ci] = thread([&, thr_ci]()
    {
        try
        {
            for (size_t i = thr_ci; i < ci; i += nthreads_ci) Ciphertext::regularize_inplace(input[i]);
        }
        catch (...)
        {
            lock_guard<mutex> lock(mtx);
            if (!exception) exception = current_exception();
        }
    });
    for (auto& thread : threads_ci) thread.join();
    if (exception) rethrow_exception(exception);

    auto shifted_input = shift_input(input, config, nthreads); // shift input

    // create mask if needed
    bool multiply = abs(multiplier - 1.0) > 10e-8;
    Plaintext weight;
    if (multiply)
    {
        const Ciphertext* reference = &shifted_input[0][0][0];
        weight = move(Plaintext(multiplier, reference->level()));
    }

    // compute average
    for (size_t thr_ci = 0; thr_ci < nthreads_ci; thr_ci++) threads_ci[thr_ci] = thread([&, thr_ci]()
    {
        try
        {
            for (size_t i = thr_ci; i < ci; i += nthreads_ci)
            {
                for (size_t k = 1; k < ksize; k++) shifted_input[0][i] += shifted_input[k][i];
                if (multiply) shifted_input[0][i] *= weight;
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
    
    return shifted_input[0];
}

} // hw

} // fhe