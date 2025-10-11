#include "auxiliary.h"

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

Tensor<3,Ciphertext> shift_input(const Tensor<2,Ciphertext>& input, const Configuration& config, size_t nthreads)
{
    const auto& kshape = config.kshape();
    const auto& mapping = config.mapping();

    const auto& ci = kshape[1];
    const auto& kh = kshape[2];
    const auto& kw = kshape[3];
    const auto& nCT = config.nct();
    const int offset = mapping[0][0];
    const size_t ksize = kh * kw;

    const size_t nthreads_ci = min(nthreads, ci);
    const size_t nthreads_hw = max(min(nthreads / nthreads_ci, ksize), 1UL);

    mutex mtx;
    exception_ptr exception;
    vector<thread> threads_ci(nthreads_ci);
    Tensor<3,Ciphertext> shifted_input{ksize, ci, nCT}; // shifted input
    for (size_t thr_ci = 0; thr_ci < nthreads_ci; thr_ci++) threads_ci[thr_ci] = thread([&, thr_ci]()
    {
        try
        {
            for (size_t i = thr_ci; i < ci; i += nthreads_ci)
            {
                vector<thread> threads_hw(nthreads_hw);
                for (size_t thr_hw = 0; thr_hw < nthreads_hw; thr_hw++) threads_hw[thr_hw] = thread([&, thr_hw]()
                {
                    try
                    {
                        for (size_t k = thr_hw; k < ksize; k += nthreads_hw)
                        {
                            size_t krow = k / kw, kcol = k % kw;
                            const int shift = int(mapping[krow][kcol]) - offset;
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

    return shifted_input;
}

} // hw

} // fhe