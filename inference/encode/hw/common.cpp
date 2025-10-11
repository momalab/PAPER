#include "common.h"

#include <algorithm>
#include <exception>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>
#include "ciphertext.h"
#include "configuration.h"
#include "defines.h"
#include "plaintext.h"
#include "tensor.h"

using namespace std;
using namespace type;

namespace fhe
{

namespace hw
{

vector<bool> pattern_output(const Configuration& config)
{
    const auto& oshape = config.oshape();
    const auto& mapping = config.omap();

    const auto& oh = oshape[1];
    const auto& ow = oshape[2];
    const auto& slots = Plaintext::default_slots();
    const auto& nCT = config.nct();
    
    vector<bool> pattern(nCT * slots, false);
    for (size_t bi = 0; bi < oh; bi++)
        for (size_t bj = 0; bj < ow; bj++)
            pattern[mapping[bi][bj]] = true;

    return pattern;
}

Tensor<2,Ciphertext>
remap
(
    const Tensor<2,Ciphertext>& input,
    const Configuration& input_config,
    const Configuration& output_config,
    const Scalar& scaling,
    double scale,
    std::size_t nthreads,
    const vector<int>& offsets
)
{
    const auto& imap = input_config.omap();
    const auto& omap = output_config.omap();
    
    if (imap.shape() != omap.shape()) throw "fhe::hw::remap: Input and output mappings are incompatible";
    if (imap == omap) return input;

    if (scale == 0.0) scale = input[0][0].keyscale();

    const auto& ci = input.size();
    const auto& nCTi = input_config.ncti();
    const auto& nCTo = output_config.ncti();
    const auto& slots = Plaintext::default_slots();

    mutex mtx;
    exception_ptr exception;

    vector<int> shifts;
    Tensor<2,Plaintext> masks;
    size_t nmasks;
    {
        // identify shift amounts and create a mask per shift
        std::map<int,vector<vector<Scalar>>> shift_map;
        for (size_t i = 0; i < imap.size(); i++)
        {
            for (size_t j = 0; j < imap[i].size(); j++)
            {
                int shift = int(imap[i][j]) - int(omap[i][j]); // positive for shift left
                if (shift_map.find(shift) == shift_map.end()) // create a mask for this shift if it doesn't exist
                    shift_map[shift] = vector<vector<double>>(nCTi, vector<double>(slots, 0.0));
                for (int offset : offsets)
                {
                    auto idx = imap[i][j] + offset;
                    shift_map[shift][idx / slots][idx % slots] = scaling;
                }
            }
        }
        for (auto& [shift, masks] : shift_map) shifts.emplace_back(shift); // copy shifts
        nmasks = shifts.size();
        masks = Tensor<2,Plaintext>{nmasks, nCTi};

        // encode masks
        size_t nthreads_sm = min(nthreads, nmasks);
        vector<thread> threads_sm(nthreads_sm);
        for (size_t thr_sm = 0; thr_sm < nthreads_sm; thr_sm++) threads_sm[thr_sm] = thread([&, thr_sm]()
        {
            for (size_t sidx = thr_sm; sidx < nmasks; sidx += nthreads_sm)
            {
                auto shift = shifts[sidx];
                auto& mask = shift_map[shift];
                for (size_t ict = 0; ict < nCTi; ict++) masks[sidx][ict] = move(Plaintext(mask[ict], input[0][ict].level()));
            }
        });
        for (auto& thread : threads_sm) thread.join();
    }
       
    // mask and shift input
    size_t nthreads_ci = min(nthreads, ci);
    size_t nthreads_sm = max(min(nthreads / nthreads_ci, nmasks), 1UL);
    Tensor<2,Ciphertext> r{ci, nCTo};
    vector<thread> threads_ci(nthreads_ci);
    for (size_t thr_ci = 0; thr_ci < nthreads_ci; thr_ci++) threads_ci[thr_ci] = thread([&, thr_ci]()
    {
        try
        {
            for (size_t i = thr_ci; i < ci; i += nthreads_ci)
            {
                Tensor<2,Ciphertext> selections{nCTo, nthreads_sm};
                vector<thread> threads_sm(nthreads_sm);
                for (size_t thr_sm = 0; thr_sm < nthreads_sm; thr_sm++) threads_sm[thr_sm] = thread([&, thr_sm, i]()
                {
                    try
                    {
                        for (size_t sidx = thr_sm; sidx < nmasks; sidx += nthreads_sm)
                        {
                            auto sel = input[i] * masks[sidx];
                            Ciphertext::regularize_inplace(sel);
                            Ciphertext::shiftleft_inplace(sel.vector(), shifts[sidx]);
                            for (size_t oct = 0; oct < nCTo; oct++)
                            {
                                if (sidx < nthreads_sm) selections[oct][thr_sm] = move(sel[oct]);
                                else selections[oct][thr_sm] += sel[oct];
                            }
                        }
                    }
                    catch (...)
                    {
                        lock_guard<mutex> lock(mtx);
                        if (!exception) exception = current_exception();
                    }
                });
                for (auto& thread : threads_sm) thread.join();
                if (exception) rethrow_exception(exception);
                for (size_t oct = 0; oct < nCTo; oct++)
                    r[i][oct] = move(Ciphertext::add_inplace(selections[oct].vector()));
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

Tensor<2,Ciphertext>
reformat
(
    const Tensor<2,Ciphertext>& output, // Co x nCT (Oh x Ow in Ih x Iw)
    const Configuration& config,
    const Scalar& scaling
)
{
    const auto& oshape = config.oshape();

    const auto& co = oshape[0];
    const auto& oh = oshape[1];
    const auto& ow = oshape[2];
    const auto& slots = Plaintext::default_slots();
    const size_t nCTo = (oh * ow) / slots + ((oh * ow) % slots != 0);

    if (co != output.size()) throw invalid_argument("fhe::hw::reformat: Invalid tensor dimensions");

    auto [indices, lengths] = Plaintext::indices_and_lengths(pattern_output(config));
    auto masks = Plaintext::create_masks(indices, lengths, scaling);

    Tensor<2,Ciphertext> r{co, nCTo};
    for (size_t o = 0; o < co; o++) // reformatting
    {
        vector<Ciphertext> v;
        for (size_t i = 0, oct = 0, oslot = 0; i < indices.size(); i++)
        {
            auto& idx = indices[i];
            auto& len = lengths[i];
            size_t ict = idx / slots;
            size_t slot = idx % slots;
            auto& mask = masks.at(slot * slots + len);
            v.emplace_back(output[o][ict] * mask);
            v.back() <<= slot - oslot;
            oslot += len;
            if (oslot == slots || i == indices.size()-1)
            {
                r[o][oct] = Ciphertext::add(v);
                r[o][oct++].regularize_inplace();
                v.clear();
                oslot = 0;
            }
        }
    }

    return r;
}

Tensor<4,Scalar> sanitize(const Tensor<4,Scalar>& kernel, const Configuration& config)
{
    const auto& ishape = config.ishape();
    const auto& kshape = config.kshape();
    const auto& stride = config.stride();
    const auto& padding = config.padding();

    const auto& ih = ishape[1];
    const auto& iw = ishape[2];
    const auto& co = kshape[0];
    const auto& ci = kshape[1];
    const auto& kh = kshape[2];
    const auto& kw = kshape[3];
    const auto& sh = stride[0];
    const auto& sw = stride[1];
    const auto& ph = padding[0];
    const auto& pw = padding[1];

    if (kernel.empty()) throw invalid_argument("fhe::hw::common::sanitize: Empty tensor");
    if (co != kernel.size()) throw invalid_argument("fhe::hw::common::sanitize: Invalid tensor output channel dimension");
    if (ci != kernel[0].size()) throw invalid_argument("fhe::hw::common::sanitize: Invalid tensor input channel dimension");
    if (kh != kernel[0][0].size()) throw invalid_argument("fhe::hw::common::sanitize: Invalid tensor kernel height dimension");
    if (kw != kernel[0][0][0].size()) throw invalid_argument("fhe::hw::common::sanitize: Invalid tensor kernel width dimension");
    if (!ph && !pw) return kernel;

    Tensor<4,Scalar> w(kernel.shape());
    for (size_t o = 0; o < co; o++)
    {
        for (size_t i = 0; i < ci; i++)
        {
            for (size_t krow = 0; krow < kh; krow++)
            {
                size_t iidx = krow;
                while (iidx < ph) iidx += sh;
                iidx -= ph;
                if (iidx >= ih || kh + iidx - krow > ih + ph) continue;
                for (size_t kcol = 0; kcol < kw; kcol++)
                {
                    size_t ijdx = kcol;
                    while (ijdx < pw) ijdx += sw;
                    ijdx -= pw;
                    if (ijdx >= iw || kw + ijdx - kcol > iw + pw) continue;
                    w[o][i][krow][kcol] = kernel[o][i][krow][kcol];
                }
            }
        }
    }
    return w;
}

} // hw

} // fhe