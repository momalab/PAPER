#include "polyskip.h"

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
polyskip
(
    Tensor<2,Ciphertext>& input1, // Ci x nCT (Ih x Iw)
    Tensor<2,Ciphertext>& input2, // Ci x nCT (Ih x Iw)
    const Tensor<4,Scalar>& kernel, // 2n x Ci x nCT x Ih x Iw
    const Configuration& config1,
    const Configuration& config2,
    size_t nthreads
)
{
    auto x1 = input1;
    auto x2 = input2;
    polyskip_inplace(x1, x2, kernel, config1, config2, nthreads);
    return x1;
}

void // Ci x nCT (Ih x Iw)
polyskip_inplace
(
    Tensor<2,Ciphertext>& input1, // Ci x nCT (Ih x Iw)
    Tensor<2,Ciphertext>& input2, // Ci x nCT (Ih x Iw)
    const Tensor<4,Scalar>& kernel, // 2n x Ci x nCT x Ih x Iw
    const Configuration& config1,
    const Configuration& config2,
    size_t nthreads
)
{
    const auto& ksize = kernel.size();
    if (ksize != 5) throw "fhe::hw::polyskip: Kernel must be at least a 5-element vector";
    
    Ciphertext::refit_inplace(input1);
    Ciphertext::refit_inplace(input2);
    int level_xa = input1[0][0].level();
    while (input2[0][0].level() > level_xa + 1) Ciphertext::modswitch_inplace(input2);
    
    int level_xb = input2[0][0].level();
    if (level_xb != level_xa + 1) throw invalid_argument("fhe::hw::polyskip: Level of input2 (" + to_string(level_xb) + ") must be level of input1 (" + to_string(level_xa) + ") + 1");

    const bool remap_needed = config1.omap() != config2.omap();
    if (remap_needed) input2 = remap(input2, config2, config1, 1.0, nthreads);

    level_xb = input2[0][0].level();
    const double keyscale = input2[0][0].keyscale();

    const auto& ci = input1.size();
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
                auto& xa = input1[i];
                auto& xb = input2[i];

                // xa^2
                auto xa2 = xa * xa;
                Ciphertext::relinearize_inplace(xa2);

                // db2 * xb^2
                Tensor<1,Ciphertext> db2xb2;
                if (remap_needed)
                {
                    auto db2 = encode_kernel(kernel, 4UL, i, config1, nthreads_ek, level_xb, keyscale);
                    db2xb2 = xb * db2;
                    db2xb2 *= xb;
                    Ciphertext::relinearize_inplace(db2xb2);
                    Ciphertext::regularize_inplace(db2xb2);
                }
                else
                {
                    db2xb2 = xb * xb;
                    Ciphertext::relinearize_inplace(db2xb2);
                    db2xb2 *= encode_kernel(kernel, 4UL, i, config1, nthreads_ek, &db2xb2[0]);
                    Ciphertext::regularize_inplace(db2xb2);
                }

                // (dab * xb) * xa
                Tensor<1,Ciphertext> dabxab;
                if (remap_needed)
                {
                    dabxab = xb * encode_kernel(kernel, 3UL, i, config1, nthreads_ek, level_xb, keyscale);
                    Ciphertext::regularize_inplace(dabxab);
                    dabxab *= xa;
                    Ciphertext::relinearize_inplace(dabxab);
                }
                else
                {
                    dabxab = xb * encode_kernel(kernel, 3UL, i, config1, nthreads_ek, &xb[0]);
                    Ciphertext::refit_inplace(dabxab);
                    dabxab *= xa;
                    Ciphertext::relinearize_inplace(dabxab);
                }
                
                // da * xa
                xa *= encode_kernel(kernel, 2UL, i, config1, nthreads_ek, &xa[0]);

                // db * xb
                if (!remap_needed) Ciphertext::modswitch_inplace(xb);
                xb *= encode_kernel(kernel, 1UL, i, config1, nthreads_ek, &xb[0]);

                // output
                xa += xb;
                xa += xa2;
                xa += dabxab;
                xa += db2xb2;
                xa += encode_kernel(kernel, 0UL, i, config1, nthreads_ek, &xa[0]); // xa is the output
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
