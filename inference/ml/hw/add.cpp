#include "add.h"

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

Tensor<2,Ciphertext> // Ci x nCT (Ih1 x Iw1)
add
(
    Tensor<2,Ciphertext>& input1, // Ci x nCT (Ih1 x Iw1)
    Tensor<2,Ciphertext>& input2, // Ci x nCT (Ih2 x Iw2)
    const Configuration& config1,
    const Configuration& config2,
    size_t nthreads,
    const vector<int>& offsets
)
{
    auto x1 = input1;
    auto x2 = input2;
    add_inplace(x1, x2, config1, config2, nthreads, offsets);
    return x1; // Return the modified input1
}

void
add_inplace
(
    Tensor<2,Ciphertext>& input1, // Ci x nCT (Ih1 x Iw1)
    Tensor<2,Ciphertext>& input2, // Ci x nCT (Ih2 x Iw2)
    const Configuration& config1,
    const Configuration& config2,
    size_t nthreads,
    const vector<int>& offsets
)
{
    // Assign the input with higher depth to yh
    Tensor<2,Ciphertext> *yl_ptr, *yh_ptr;
    const Configuration *configl_ptr, *configh_ptr;
    {
        int level1 = input1[0][0].level();
        int level2 = input2[0][0].level();
        double scale1 = input1[0][0].scale();
        double scale2 = input2[0][0].scale();
        yh_ptr = &input1;
        yl_ptr = &input2;
        configh_ptr = &config1;
        configl_ptr = &config2;
        if ((level1 < level2) || (level1 == level2 && scale1 > scale2))
        {
            swap(yl_ptr, yh_ptr);
            swap(configl_ptr, configh_ptr);
        }
    }
    Tensor<2,Ciphertext>& yl = *yl_ptr;
    Tensor<2,Ciphertext>& yh = *yh_ptr;

    const bool remap_needed = configl_ptr->omap() != configh_ptr->omap();

    int level_yl = yl[0][0].level();
    double scale_yl = yl[0][0].scale();
    double scale_yh = yh[0][0].scale();
    double keyscale = yh[0][0].keyscale();
    double logscale_key= log2(keyscale);
    double logscale_yl = log2(scale_yl);
    double logscale_yh = log2(scale_yh);
    int target_level = level_yl + int(remap_needed || round((logscale_yh - logscale_yl) / logscale_key) > 0);
    mutex mtx;
    exception_ptr exception;
    const auto& ci = input1.size();
    size_t nthreads_ci = min(nthreads, ci);
    vector<thread> threads_ci(nthreads_ci);
    for (size_t thr_ci = 0; thr_ci < nthreads_ci; thr_ci++) threads_ci[thr_ci] = thread([&, thr_ci]()
    {
        try
        {
            for (size_t i = thr_ci; i < ci; i += nthreads_ci) Ciphertext::modswitch_inplace(yh[i], target_level);
        }
        catch (...)
        {
            lock_guard<mutex> lock(mtx);
            if (!exception) exception = current_exception();
        }
    });
    for (auto& thread : threads_ci) thread.join();
    if (exception) rethrow_exception(exception);
    
    int level_yh = yh[0][0].level();
    scale_yh = yh[0][0].scale();
    logscale_yh = log2(scale_yh);
    bool same_level = level_yh == level_yl;
    bool same_sublevel = same_level && round((logscale_yh - logscale_yl) / logscale_key) == 0;

    double scale = same_sublevel ? keyscale : (same_level ? 1.0 : double(yh[0][0].qi())) * scale_yl / scale_yh;
    if (remap_needed) yh = remap(yh, *configh_ptr, *configl_ptr, 1.0, scale, nthreads, offsets);

    if (remap_needed == same_sublevel)
    {
        Plaintext one(1.0, level_yh, scale);
        for (size_t thr_ci = 0; thr_ci < nthreads_ci; thr_ci++) threads_ci[thr_ci] = thread([&, thr_ci]()
        {
            try
            {
                for (size_t i = thr_ci; i < ci; i += nthreads_ci)
                {
                    if (same_sublevel) yl[i] *= one;
                    else
                    {
                        yh[i] *= one;
                        if (!same_level) Ciphertext::regularize_inplace(yh[i]);
                    }
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

    input1 += input2;
}

} // hw

} // fhe