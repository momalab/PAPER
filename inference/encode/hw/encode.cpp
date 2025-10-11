#include "encode.h"

#include <array>
#include <exception>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>
#include "ciphertext.h"
#include "configuration.h"
#include "defines.h"
#include "hash.h"
#include "plaintext.h"
#include "tensor.h"
#include "transform.h"

using namespace std;
using namespace type;

namespace fhe
{

namespace hw
{

Tensor<4,Scalar> decode_kernel(const Tensor<5,Plaintext>& kernel, const Configuration& config, size_t nthreads, int offset)
{
    auto ks = transform<6,Scalar,5,Plaintext>(kernel, nthreads); // decoding
    return unmap_kernel(ks, config, offset); // unmapping
}

Tensor<5,Scalar> decode_kernel(const Tensor<5,Plaintext>& kernel, const Configuration& config, size_t nthreads, const vector<int>& offsets)
{
    auto ks = transform<6,Scalar,5,Plaintext>(kernel, nthreads); // decoding
    return unmap_kernel(ks, config, offsets); // unmapping
}

Tensor<5,Plaintext> encode_kernel(const Tensor<4,Scalar>& kernel, const Configuration& config, size_t nthreads, const Ciphertext* reference)
{
    auto mapped = map_kernel(kernel, config);
    return transform<5,Plaintext,6,Scalar>(mapped, nthreads, reference); // encoding
}

Tensor<5,Plaintext> encode_kernel(const Tensor<4,Scalar>& kernel, const Configuration& config, size_t nthreads, int level, double scale)
{
    auto mapped = map_kernel(kernel, config);
    return transform<5,Plaintext,6,Scalar>(mapped, nthreads, level, scale); // encoding
}

Tensor<5,Plaintext> encode_kernel(const Tensor<6,Scalar>& kernel, const Configuration& config, size_t nthreads, const Ciphertext* reference)
{
    auto mapped = map_kernel(kernel, config, nthreads);
    return transform<5,Plaintext,6,Scalar>(mapped, nthreads, reference); // encoding
}

Tensor<5,Plaintext> encode_kernel(const Tensor<6,Scalar>& kernel, const Configuration& config, size_t nthreads, int level, double scale)
{
    auto mapped = map_kernel(kernel, config, nthreads);
    return transform<5,Plaintext,6,Scalar>(mapped, nthreads, level, scale); // encoding
}

vector<KernelMap> encode_kernel_map(const Tensor<6,Scalar>& kernel, const Configuration& config, size_t nthreads, int level, double scale)
{
    const auto& kshape = config.kshape();

    const auto& co = kshape[0];
    const auto& ci = kshape[1];
    const auto& kh = kshape[2];
    const auto& kw = kshape[3];
    const size_t ksize = kh * kw;
    
    vector<unordered_map
    <
        Scalar,
        pair<array<int,3>, Tensor<2,array<size_t,2>>>
    >> unique_weights(kw);
    for (size_t o = 0; o < co; o++)
    {
        for (size_t k = 0; k < ksize; k++)
        {
            size_t krow = k / kw, kcol = k % kw;
            auto& map = unique_weights[kcol];
            for (size_t i = 0; i < ci; i++)
            {
                Scalar key = kernel[o][krow][kcol][i][0][0];
                if (map.find(key) == map.end()) // encode value into map
                {
                    // set encoding parameters and initialize Tensor of indices
                    int shift_i = config.shift(     0, kcol);
                    int shift_f = config.shift(kh - 1, kcol);
                    map[key] = make_pair(array{int(kcol), shift_i, shift_f}, Tensor<2,array<size_t,2>>{co, 0});
                }
                // append the indices of the kernel value
                auto& pos = map[key].second;
                pos[o].push_back({krow, i});
            }
        }
    }

    // convert unique weights from unordered_map to vectors for parallel encoding
    Tensor<2,double> unique_keys{kw, 0};
    Tensor<2,array<int,3>> positions{kw, 0};
    for (size_t k = 0; k < kw; k++)
    {
        auto& keys = unique_keys[k];
        auto& poss = positions[k];
        for (const auto& [key, value] : unique_weights[k])
        {
            keys.push_back(key);
            poss.push_back(value.first); // position (kcol, shift_i, shift_f)
        }
    }
    
    // encode unique weights into Plaintext tensors
    mutex mtx;
    exception_ptr exception;
    size_t nthreads_hw = max(min(nthreads, kw), 1UL);
    vector<KernelMap> kernel_maps(kw);
    vector<thread> threads_hw(nthreads_hw);
    for (size_t thr_hw = 0; thr_hw < nthreads_hw; thr_hw++) threads_hw[thr_hw] = thread([&, thr_hw]()
    {
        try
        {
            for (size_t k = thr_hw; k < kw; k += nthreads_hw)
            {
                auto& kmap = kernel_maps[k];
                auto& keys = unique_keys[k];
                auto& poss = positions[k];
                const size_t& nkeys = keys.size();
                Tensor<2,Plaintext> encoded_weights{nkeys, 0};
                size_t nthreads_keys = max(min(nthreads / nthreads_hw, nkeys), 1UL);
                size_t nthreads_encode = nthreads / (nthreads_hw * nthreads_keys);
                vector<thread> threads_keys(nthreads_keys);
                for (size_t thr_key = 0; thr_key < nthreads_keys; thr_key++) threads_keys[thr_key] = thread([&, thr_key]()
                {
                    try
                    {
                        for (size_t i = thr_key; i < nkeys; i += nthreads_keys)
                        {
                            auto& key = keys[i];
                            auto& pos = poss[i];
                            int kcol = pos[0], shift_i = pos[1], shift_f = pos[2];
                            auto wi = map_weight(key,    0, kcol, shift_i, config);
                            auto wf = map_weight(key, kh-1, kcol, shift_f, config);
                            auto shape = wi.shape();
                            for (size_t ii = 0; ii < shape[0]; ii++)
                                for (size_t ij = 0; ij < shape[1]; ij++)
                                    wi[ii][ij] = abs(wi[ii][ij]) > abs(wf[ii][ij]) ? wi[ii][ij] : wf[ii][ij];
                            encoded_weights[i] = transform<1,Plaintext,2,Scalar>(wi, nthreads_encode, level, scale);
                        }
                    }
                    catch (...)
                    {
                        lock_guard<mutex> lock(mtx);
                        if (!exception) exception = current_exception();
                    }
                });
                for (auto& thread : threads_keys) thread.join();
                if (exception) rethrow_exception(exception);

                // move encoded weights into the unordered_map
                for (size_t i = 0; i < nkeys; i++)
                {
                    auto& key = keys[i];
                    auto& indices = unique_weights[k][key].second; // get the indices from the original map
                    pair<Tensor<1,Plaintext>, Tensor<2,array<size_t,2>>> value{move(encoded_weights[i]), move(indices)};
                    kmap.emplace(key, move(value)); // insert into the kernel map
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
    
    return kernel_maps;
}

KernelMap encode_kernel_map(const Tensor<6,Scalar>& kernel, const Configuration& config, size_t nthreads, int level, double scale, size_t kcol)
{
    const auto& kshape = config.kshape();

    const auto& co = kshape[0];
    const auto& ci = kshape[1];
    const auto& kh = kshape[2];
    
    unordered_map
    <
        Scalar,
        pair<array<int,3>, Tensor<2,array<size_t,2>>>
    > unique_weights;
    for (size_t o = 0; o < co; o++)
    {
        for (size_t krow = 0; krow < kh; krow++)
        {
            auto& map = unique_weights;
            for (size_t i = 0; i < ci; i++)
            {
                Scalar key = kernel[o][krow][kcol][i][0][0];
                if (map.find(key) == map.end()) // encode value into map
                {
                    // set encoding parameters and initialize Tensor of indices
                    int shift_i = config.shift(     0, kcol);
                    int shift_f = config.shift(kh - 1, kcol);
                    map[key] = make_pair(array{int(kcol), shift_i, shift_f}, Tensor<2,array<size_t,2>>{co, 0});
                }
                // append the indices of the kernel value
                auto& pos = map[key].second;
                pos[o].push_back({krow, i});
            }
        }
    }

    // convert unique weights from unordered_map to vectors for parallel encoding
    Tensor<1,double> unique_keys;
    Tensor<1,array<int,3>> positions;
    {
        auto& keys = unique_keys;
        auto& poss = positions;
        for (const auto& [key, value] : unique_weights)
        {
            keys.push_back(key);
            poss.push_back(value.first); // position (kcol, shift_i, shift_f)
        }
    }
    
    // encode unique weights into Plaintext tensors
    mutex mtx;
    exception_ptr exception;
    KernelMap kernel_map;
    {
        auto& kmap = kernel_map;
        auto& keys = unique_keys;
        auto& poss = positions;
        const size_t& nkeys = keys.size();
        Tensor<2,Plaintext> encoded_weights{nkeys, 0};
        size_t nthreads_keys = max(min(nthreads, nkeys), 1UL);
        size_t nthreads_encode = nthreads / nthreads_keys;
        vector<thread> threads_keys(nthreads_keys);
        for (size_t thr_key = 0; thr_key < nthreads_keys; thr_key++) threads_keys[thr_key] = thread([&, thr_key]()
        {
            try
            {
                for (size_t i = thr_key; i < nkeys; i += nthreads_keys)
                {
                    auto& key = keys[i];
                    auto& pos = poss[i];
                    int kcol = pos[0], shift_i = pos[1], shift_f = pos[2];
                    auto wi = map_weight(key,    0, kcol, shift_i, config);
                    auto wf = map_weight(key, kh-1, kcol, shift_f, config);
                    auto shape = wi.shape();
                    for (size_t ii = 0; ii < shape[0]; ii++)
                        for (size_t ij = 0; ij < shape[1]; ij++)
                            wi[ii][ij] = abs(wi[ii][ij]) > abs(wf[ii][ij]) ? wi[ii][ij] : wf[ii][ij];
                    encoded_weights[i] = transform<1,Plaintext,2,Scalar>(wi, nthreads_encode, level, scale);
                }
            }
            catch (...)
            {
                lock_guard<mutex> lock(mtx);
                if (!exception) exception = current_exception();
            }
        });
        for (auto& thread : threads_keys) thread.join();
        if (exception) rethrow_exception(exception);

        // move encoded weights into the unordered_map
        for (size_t i = 0; i < nkeys; i++)
        {
            auto& key = keys[i];
            auto& indices = unique_weights[key].second; // get the indices from the original map
            pair<Tensor<1,Plaintext>, Tensor<2,array<size_t,2>>> value{move(encoded_weights[i]), move(indices)};
            kmap.emplace(key, move(value)); // insert into the kernel map
        }
    }
    
    return kernel_map;
}

vector<KernelEnsembleMap> encode_kernel_map(const Tensor<7,Scalar>& kernel, const Configuration& config, size_t nthreads, int level, double scale, const vector<int>& offsets)
{
    const auto& kshape = config.kshape();

    const auto& co = kshape[0];
    const auto& ci = kshape[1];
    const auto& kh = kshape[2];
    const auto& kw = kshape[3];
    const size_t ksize = kh * kw;
    const auto& nModels = kernel.size();
    
    vector<unordered_map
    <
        vector<Scalar>,
        pair<array<int,3>, Tensor<2,array<size_t,2>>>,
        util::VectorDoubleHash,
        util::VectorDoubleEqual
    >> unique_weights(kw);
    for (size_t o = 0; o < co; o++)
    {
        for (size_t k = 0; k < ksize; k++)
        {
            size_t krow = k / kw, kcol = k % kw;
            auto& map = unique_weights[kcol];
            for (size_t i = 0; i < ci; i++)
            {
                vector<Scalar> key;
                for (size_t model_idx = 0; model_idx < nModels; model_idx++) key.push_back(kernel[model_idx][o][krow][kcol][i][0][0]);

                if (map.find(key) == map.end()) // encode value into map
                {
                    // set encoding parameters and initialize Tensor of indices
                    int shift_i = config.shift(     0, kcol);
                    int shift_f = config.shift(kh - 1, kcol);
                    map[key] = make_pair(array{int(kcol), shift_i, shift_f}, Tensor<2,array<size_t,2>>{co, 0});
                }
                // append the indices of the kernel value
                auto& pos = map[key].second;
                pos[o].push_back({krow, i});
            }
        }
    }

    // convert unique weights from unordered_map to vectors for parallel encoding
    Tensor<2,vector<double>> unique_keys{kw, 0};
    Tensor<2,array<int,3>> positions{kw, 0};
    for (size_t k = 0; k < kw; k++)
    {
        auto& keys = unique_keys[k];
        auto& poss = positions[k];
        for (const auto& [key, value] : unique_weights[k])
        {
            keys.push_back(key);
            poss.push_back(value.first); // position (kcol, shift_i, shift_f)
        }
    }
    
    // encode unique weights into Plaintext tensors
    mutex mtx;
    exception_ptr exception;
    size_t nthreads_hw = max(min(nthreads, kw), 1UL);
    vector<KernelEnsembleMap> kernel_maps(kw);
    vector<thread> threads_hw(nthreads_hw);
    for (size_t thr_hw = 0; thr_hw < nthreads_hw; thr_hw++) threads_hw[thr_hw] = thread([&, thr_hw]()
    {
        try
        {
            for (size_t k = thr_hw; k < kw; k += nthreads_hw)
            {
                auto& kmap = kernel_maps[k];
                auto& keys = unique_keys[k];
                auto& poss = positions[k];
                const size_t& nkeys = keys.size();
                Tensor<2,Plaintext> encoded_weights{nkeys, 0};
                size_t nthreads_keys = max(min(nthreads / nthreads_hw, nkeys), 1UL);
                size_t nthreads_encode = nthreads / (nthreads_hw * nthreads_keys);
                vector<thread> threads_keys(nthreads_keys);
                for (size_t thr_key = 0; thr_key < nthreads_keys; thr_key++) threads_keys[thr_key] = thread([&, thr_key]()
                {
                    try
                    {
                        for (size_t i = thr_key; i < nkeys; i += nthreads_keys)
                        {
                            auto& key = keys[i];
                            auto& pos = poss[i];
                            int kcol = pos[0], shift_i = pos[1], shift_f = pos[2];
                            auto wi = map_weight(key,    0, kcol, shift_i, config, offsets);
                            auto wf = map_weight(key, kh-1, kcol, shift_f, config, offsets);
                            auto shape = wi.shape();
                            for (size_t ii = 0; ii < shape[0]; ii++)
                                for (size_t ij = 0; ij < shape[1]; ij++)
                                    wi[ii][ij] = abs(wi[ii][ij]) > abs(wf[ii][ij]) ? wi[ii][ij] : wf[ii][ij];
                            encoded_weights[i] = transform<1,Plaintext,2,Scalar>(wi, nthreads_encode, level, scale);
                        }
                    }
                    catch (...)
                    {
                        lock_guard<mutex> lock(mtx);
                        if (!exception) exception = current_exception();
                    }
                });
                for (auto& thread : threads_keys) thread.join();
                if (exception) rethrow_exception(exception);

                // move encoded weights into the unordered_map
                for (size_t i = 0; i < nkeys; i++)
                {
                    auto& key = keys[i];
                    auto& indices = unique_weights[k][key].second; // get the indices from the original map
                    pair<Tensor<1,Plaintext>, Tensor<2,array<size_t,2>>> value{move(encoded_weights[i]), move(indices)};
                    kmap.emplace(key, move(value)); // insert into the kernel map
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
    
    return kernel_maps;
}

KernelEnsembleMap encode_kernel_map(const Tensor<7,Scalar>& kernel, const Configuration& config, size_t nthreads, int level, double scale, const vector<int>& offsets, size_t kcol)
{
    const auto& kshape = config.kshape();

    const auto& co = kshape[0];
    const auto& ci = kshape[1];
    const auto& kh = kshape[2];
    const auto& nModels = kernel.size();
    
    unordered_map
    <
        vector<Scalar>,
        pair<array<int,3>, Tensor<2,array<size_t,2>>>,
        util::VectorDoubleHash,
        util::VectorDoubleEqual
    > unique_weights;
    for (size_t o = 0; o < co; o++)
    {
        for (size_t krow = 0; krow < kh; krow++)
        {
            auto& map = unique_weights;
            for (size_t i = 0; i < ci; i++)
            {
                vector<Scalar> key;
                for (size_t model_idx = 0; model_idx < nModels; model_idx++) key.push_back(kernel[model_idx][o][krow][kcol][i][0][0]);

                if (map.find(key) == map.end()) // encode value into map
                {
                    // set encoding parameters and initialize Tensor of indices
                    int shift_i = config.shift(     0, kcol);
                    int shift_f = config.shift(kh - 1, kcol);
                    map[key] = make_pair(array{int(kcol), shift_i, shift_f}, Tensor<2,array<size_t,2>>{co, 0});
                }
                // append the indices of the kernel value
                auto& pos = map[key].second;
                pos[o].push_back({krow, i});
            }
        }
    }

    // convert unique weights from unordered_map to vectors for parallel encoding
    Tensor<1,vector<double>> unique_keys;
    Tensor<1,array<int,3>> positions;
    {
        auto& keys = unique_keys;
        auto& poss = positions;
        for (const auto& [key, value] : unique_weights)
        {
            keys.push_back(key);
            poss.push_back(value.first); // position (kcol, shift_i, shift_f)
        }
    }
    
    // encode unique weights into Plaintext tensors
    mutex mtx;
    exception_ptr exception;
    KernelEnsembleMap kernel_map;
    {
        auto& kmap = kernel_map;
        auto& keys = unique_keys;
        auto& poss = positions;
        const size_t& nkeys = keys.size();
        Tensor<2,Plaintext> encoded_weights{nkeys, 0};
        size_t nthreads_keys = max(min(nthreads, nkeys), 1UL);
        size_t nthreads_encode = nthreads / nthreads_keys;
        vector<thread> threads_keys(nthreads_keys);
        for (size_t thr_key = 0; thr_key < nthreads_keys; thr_key++) threads_keys[thr_key] = thread([&, thr_key]()
        {
            try
            {
                for (size_t i = thr_key; i < nkeys; i += nthreads_keys)
                {
                    auto& key = keys[i];
                    auto& pos = poss[i];
                    int kcol = pos[0], shift_i = pos[1], shift_f = pos[2];
                    auto wi = map_weight(key,    0, kcol, shift_i, config, offsets);
                    auto wf = map_weight(key, kh-1, kcol, shift_f, config, offsets);
                    auto shape = wi.shape();
                    for (size_t ii = 0; ii < shape[0]; ii++)
                        for (size_t ij = 0; ij < shape[1]; ij++)
                            wi[ii][ij] = abs(wi[ii][ij]) > abs(wf[ii][ij]) ? wi[ii][ij] : wf[ii][ij];
                    encoded_weights[i] = transform<1,Plaintext,2,Scalar>(wi, nthreads_encode, level, scale);
                }
            }
            catch (...)
            {
                lock_guard<mutex> lock(mtx);
                if (!exception) exception = current_exception();
            }
        });
        for (auto& thread : threads_keys) thread.join();
        if (exception) rethrow_exception(exception);

        // move encoded weights into the unordered_map
        for (size_t i = 0; i < nkeys; i++)
        {
            auto& key = keys[i];
            auto& indices = unique_weights[key].second; // get the indices from the original map
            pair<Tensor<1,Plaintext>, Tensor<2,array<size_t,2>>> value{move(encoded_weights[i]), move(indices)};
            kmap.emplace(key, move(value)); // insert into the kernel map
        }
    }
    
    return kernel_map;
}

Tensor<1,Plaintext> encode_weight(const Scalar& weight, size_t krow, size_t kcol, int leftshift, const Configuration& config, size_t nthreads, const Ciphertext* reference)
{
    auto mapped = map_weight(weight, krow, kcol, leftshift, config);
    return transform<1,Plaintext,2,Scalar>(mapped, nthreads, reference); // encoding
}

Tensor<1,Plaintext> encode_weight(const Scalar& weight, size_t krow, size_t kcol, int leftshift, const Configuration& config, size_t nthreads, int level, double scale)
{
    auto k = map_weight(weight, krow, kcol, leftshift, config);
    return transform<1,Plaintext,2,Scalar>(k, nthreads, level, scale); // encoding
}

Tensor<1,Plaintext> encode_weight(const Tensor<2,Scalar>& weight, size_t krow, size_t kcol, int leftshift, const Configuration& config, size_t nthreads, const Ciphertext* reference)
{
    auto mapped = map_weight(weight, krow, kcol, leftshift, config);
    return transform<1,Plaintext,2,Scalar>(mapped, nthreads, reference); // encoding
}

Tensor<1,Plaintext> encode_weight(const Tensor<2,Scalar>& weight, size_t krow, size_t kcol, int leftshift, const Configuration& config, size_t nthreads, int level, double scale)
{
    auto k = map_weight(weight, krow, kcol, leftshift, config);
   return transform<1,Plaintext,2,Scalar>(k, nthreads, level, scale); // encoding
}

Tensor<1,Plaintext> encode_weight(const vector<Scalar>& weight, size_t krow, size_t kcol, int leftshift, const Configuration& config, size_t nthreads, const Ciphertext* reference, const vector<int>& offsets)
{
    auto mapped = map_weight(weight, krow, kcol, leftshift, config, offsets);
    return transform<1,Plaintext,2,Scalar>(mapped, nthreads, reference); // encoding
}

Tensor<1,Plaintext> encode_weight(const vector<Scalar>& weight, size_t krow, size_t kcol, int leftshift, const Configuration& config, size_t nthreads, int level, double scale, const vector<int>& offsets)
{
    auto k = map_weight(weight, krow, kcol, leftshift, config, offsets);
    return transform<1,Plaintext,2,Scalar>(k, nthreads, level, scale); // encoding
}

Tensor<2,Scalar> map(const Tensor<2,Scalar>& input, const Configuration& config, bool is_kernel)
{
    const auto& oshape = is_kernel ? config.ishape() : config.oshape();
    const auto& mapping = is_kernel ? config.imap() : config.omap();

    const auto& oh = oshape[1];
    const auto& ow = oshape[2];
    const auto& slots = Plaintext::default_slots();
    const auto& nCT = is_kernel ? config.ncti() : config.ncto();

    Tensor<2,Scalar> mapped{nCT, slots};
    for (size_t bi = 0; bi < oh; bi++)
    {
        for (size_t bj = 0; bj < ow; bj++)
        {
            size_t idx = mapping[bi][bj];
            mapped[idx / slots][idx % slots] = input[bi][bj];
        }
    }
    return mapped;
}

Tensor<3,Scalar> map(const Tensor<3,Scalar>& input, const Configuration& config, bool is_kernel)
{
    const auto& oshape = is_kernel ? config.ishape() : config.oshape();
    const auto& kshape = config.kshape();
    const auto& co = is_kernel ? kshape[1] : oshape[0];

    Tensor<3,Scalar> mapped;
    for (size_t o = 0; o < co; o++) mapped.emplace_back(map(input[o], config, is_kernel));
    return mapped;
}

Tensor<2,Scalar> map(const Tensor<3,Scalar>& input, size_t o, const Configuration& config, bool is_kernel)
{
    const auto& oshape = is_kernel ? config.ishape() : config.oshape();
    const auto& mapping = is_kernel ? config.imap() : config.omap();

    const auto& oh = oshape[1];
    const auto& ow = oshape[2];
    const auto& slots = Plaintext::default_slots();
    const auto& nCT = is_kernel ? config.ncti() : config.ncto();

    Tensor<2,Scalar> mapped{nCT, slots};
    for (size_t bi = 0; bi < oh; bi++)
    {
        for (size_t bj = 0; bj < ow; bj++)
        {
            size_t idx = mapping[bi][bj];
            mapped[idx / slots][idx % slots] = input[o][bi][bj];
        }
    }
    return mapped;
}

Tensor<2,Scalar> map(const Tensor<4,Scalar>& input, size_t o, size_t i, const Configuration& config, bool is_kernel)
{
    const auto& oshape = is_kernel ? config.ishape() : config.oshape();
    const auto& mapping = is_kernel ? config.imap() : config.omap();

    const auto& oh = oshape[1];
    const auto& ow = oshape[2];
    const auto& slots = Plaintext::default_slots();
    const auto& nCT = is_kernel ? config.ncti() : config.ncto();

    Tensor<2,Scalar> mapped{nCT, slots};
    for (size_t bi = 0; bi < oh; bi++)
    {
        for (size_t bj = 0; bj < ow; bj++)
        {
            size_t idx = mapping[bi][bj];
            mapped[idx / slots][idx % slots] = input[o][i][bi][bj];
        }
    }
    return mapped;
}

Tensor<3,Scalar> map(const Tensor<4,Scalar>& input, const Configuration& config, bool is_kernel, const std::vector<int>& offsets)
{
    const auto& oshape = is_kernel ? config.ishape() : config.oshape();
    const auto& kshape = config.kshape();
    const auto& co = is_kernel ? kshape[1] : oshape[0];

    Tensor<3,Scalar> mapped;
    for (size_t o = 0; o < co; o++) mapped.emplace_back(map(input, o, config, is_kernel, offsets));
    return mapped;
}

Tensor<2,Scalar> map(const Tensor<4,Scalar>& input, size_t o, const Configuration& config, bool is_kernel, const vector<int>& offsets)
{
    const auto& oshape = is_kernel ? config.ishape() : config.oshape();
    const auto& mapping = is_kernel ? config.imap() : config.omap();

    const auto& oh = oshape[1];
    const auto& ow = oshape[2];
    const auto& slots = Plaintext::default_slots();
    const auto& nCT = is_kernel ? config.ncti() : config.ncto();
    const auto& nModels = offsets.size();

    Tensor<2,Scalar> mapped{nCT, slots};
    for (size_t bi = 0; bi < oh; bi++)
    {
        for (size_t bj = 0; bj < ow; bj++)
        {
            for (size_t model_idx = 0; model_idx < nModels; model_idx++)
            {
                int offset = offsets[model_idx];
                size_t idx = mapping[bi][bj] + offset;
                mapped[idx / slots][idx % slots] = input[model_idx][o][bi][bj];
            }
        }
    }
    return mapped;
}

Tensor<2,Scalar> map(const Tensor<5,Scalar>& kernel, size_t o, size_t i, const Configuration& config, bool is_kernel, const vector<int>& offsets)
{
    const auto& oshape = is_kernel ? config.ishape() : config.oshape();
    const auto& mapping = is_kernel ? config.imap() : config.omap();

    const auto& oh = oshape[1];
    const auto& ow = oshape[2];
    const auto& slots = Plaintext::default_slots();
    const auto& nCT = is_kernel ? config.ncti() : config.ncto();
    const auto& nModels = offsets.size();

    Tensor<2,Scalar> mapped{nCT, slots};
    for (size_t bi = 0; bi < oh; bi++)
    {
        for (size_t bj = 0; bj < ow; bj++)
        {
            for (size_t model_idx = 0; model_idx < nModels; model_idx++)
            {
                int offset = offsets[model_idx];
                size_t idx = mapping[bi][bj] + offset;
                mapped[idx / slots][idx % slots] = kernel[model_idx][o][i][bi][bj];
            }
        }
    }
    return mapped;
}

Tensor<6,Scalar> map_kernel(const Tensor<4,Scalar>& kernel, const Configuration& config)
{
    const auto& ishape = config.ishape();
    const auto& kshape = config.kshape();
    const auto& oshape = config.oshape();
    const auto& stride = config.stride();
    const auto& padding = config.padding();
    const auto& mapping = config.imap();

    const auto& ih = ishape[1];
    const auto& iw = ishape[2];
    const auto& ci = kernel[0][0][0].size();
    const auto& kh = kshape[2];
    const auto& kw = kshape[3];
    const auto& sh = stride[0];
    const auto& sw = stride[1];
    const auto& ph = padding[0];
    const auto& pw = padding[1];
    const auto& co = kernel.size();
    const auto& oh = oshape[1];
    const auto& ow = oshape[2];
    const auto& slots = Plaintext::default_slots();
    const auto& nCT = config.nct();

    Tensor<6,Scalar> k{co, kh, kw, ci, nCT, slots};
    for (size_t o = 0; o < co; o++) // mapping
    {
        for (size_t krow = 0; krow < kh; krow++)
        {
            for (size_t kcol = 0; kcol < kw; kcol++)
            {
                for (size_t ii = 0, bi = 0; bi < oh; ii += sh, bi++)
                {
                    for (size_t i = 0; i < ci; i++)
                    {
                        size_t iidx = ii + krow;
                        if (iidx < ph) continue;
                        iidx -= ph;
                        if (iidx >= ih) break;
                        for (size_t ij = 0, bj = 0; bj < ow; ij += sw, bj++)
                        {
                            size_t ijdx = ij + kcol;
                            if (ijdx < pw) continue;
                            ijdx -= pw;
                            if (ijdx >= iw) break;
                            size_t idx = mapping[iidx][ijdx];
                            if (idx / slots >= nCT) break;
                            k[o][krow][kcol][i][idx / slots][idx % slots] = kernel[o][krow][kcol][i];
                        }
                    }
                }
            }
        }
    }
    return k;
}

Tensor<6,Scalar> map_kernel(const Tensor<6,Scalar>& kernel, const Configuration& config, size_t nthreads)
{
    const auto& kshape = config.kshape();
    const auto& oshape = config.oshape();
    const auto& padding = config.padding();
    const auto& mapping = config.mapping();

    const auto& ci = kshape[1];
    const auto& kh = kshape[2];
    const auto& kw = kshape[3];
    const auto& co = oshape[0];
    const auto& ph = padding[0];
    const auto& pw = padding[1];
    const auto& slots = Plaintext::default_slots();
    const auto& nCT = config.nct();
    const int offset = mapping[ph][pw];
    const size_t ksize = kh * kw;

    mutex mtx;
    exception_ptr exception;
    size_t nthreads_co = max(min(co, nthreads), 1UL);
    size_t nthreads_hw = max(min(ksize, nthreads / nthreads_co), 1UL);

    Tensor<6,Scalar> w{co, kh, kw, ci, nCT, slots};
    vector<thread> threads_co(nthreads_co);
    for (size_t thr_co = 0; thr_co < nthreads_co; thr_co++) threads_co[thr_co] = thread([&, thr_co]()
    {
        try
        {
            for (size_t o = thr_co; o < co; o += nthreads_co)
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
                            for (size_t i = 0; i < ci; i++) w[o][krow][kcol][i] = map_weight(kernel[o][krow][kcol][i], krow, kcol, shift, config);
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
    for (auto& thread : threads_co) thread.join();
    if (exception) rethrow_exception(exception);
    
    return w;
}

Tensor<2,Scalar> map_weight(const Scalar& weight, size_t krow, size_t kcol, int leftshift, const Configuration& config)
{
    const auto& ishape = config.ishape();
    const auto& oshape = config.oshape();
    const auto& stride = config.stride();
    const auto& padding = config.padding();
    const auto& mapping = config.imap();

    const auto& ih = ishape[1];
    const auto& iw = ishape[2];
    const auto& sh = stride[0];
    const auto& sw = stride[1];
    const auto& ph = padding[0];
    const auto& pw = padding[1];
    const auto& oh = oshape[1];
    const auto& ow = oshape[2];
    const auto& slots = Plaintext::default_slots();
    const auto& nCT = config.nct();

    Tensor<2,Scalar> k{nCT, slots};
    for (size_t ii = 0, bi = 0; bi < oh; ii += sh, bi++)
    {
        size_t iidx = ii + krow;
        if (iidx < ph) continue;
        iidx -= ph;
        if (iidx >= ih) break;
        for (size_t ij = 0, bj = 0; bj < ow; ij += sw, bj++)
        {
            size_t ijdx = ij + kcol;
            if (ijdx < pw) continue;
            ijdx -= pw;
            if (ijdx >= iw) break;
            int idx = mapping[iidx][ijdx];
            if (idx / slots >= nCT) break;
            idx -= leftshift;
            if (idx < 0) continue;
            k[idx / slots][idx % slots] = weight;
        }
    }
    return k;
}

Tensor<2,Scalar> map_weight(const Tensor<2,Scalar>& weight, size_t krow, size_t kcol, int leftshift, const Configuration& config)
{
    const auto& ishape = config.ishape();
    const auto& oshape = config.oshape();
    const auto& stride = config.stride();
    const auto& padding = config.padding();
    const auto& mapping = config.imap();

    const auto& ih = ishape[1];
    const auto& iw = ishape[2];
    const auto& sh = stride[0];
    const auto& sw = stride[1];
    const auto& ph = padding[0];
    const auto& pw = padding[1];
    const auto& oh = oshape[1];
    const auto& ow = oshape[2];
    const auto& slots = Plaintext::default_slots();
    const auto& nCT = config.nct();

    Tensor<2,Scalar> k{nCT, slots};
    for (size_t ii = 0, bi = 0; bi < oh; ii += sh, bi++)
    {
        size_t iidx = ii + krow;
        if (iidx < ph) continue;
        iidx -= ph;
        if (iidx >= ih) break;
        for (size_t ij = 0, bj = 0; bj < ow; ij += sw, bj++)
        {
            size_t ijdx = ij + kcol;
            if (ijdx < pw) continue;
            ijdx -= pw;
            if (ijdx >= iw) break;
            int idx = mapping[iidx][ijdx];
            if (idx / slots >= nCT) break;
            idx -= leftshift;
            if (idx < 0) continue;
            k[idx / slots][idx % slots] = weight[ii][ij];
        }
    }
    return k;
}

Tensor<2,Scalar> map_weight(const vector<Scalar>& weight, size_t krow, size_t kcol, int leftshift, const Configuration& config, const std::vector<int>& offsets)
{
    const auto& ishape = config.ishape();
    const auto& oshape = config.oshape();
    const auto& stride = config.stride();
    const auto& padding = config.padding();
    const auto& mapping = config.imap();

    const auto& ih = ishape[1];
    const auto& iw = ishape[2];
    const auto& sh = stride[0];
    const auto& sw = stride[1];
    const auto& ph = padding[0];
    const auto& pw = padding[1];
    const auto& oh = oshape[1];
    const auto& ow = oshape[2];
    const auto& slots = Plaintext::default_slots();
    const auto& nCT = config.nct();
    const auto& nModels = offsets.size();

    Tensor<2,Scalar> k{nCT, slots};
    for (size_t ii = 0, bi = 0; bi < oh; ii += sh, bi++)
    {
        size_t iidx = ii + krow;
        if (iidx < ph) continue;
        iidx -= ph;
        if (iidx >= ih) break;
        for (size_t ij = 0, bj = 0; bj < ow; ij += sw, bj++)
        {
            size_t ijdx = ij + kcol;
            if (ijdx < pw) continue;
            ijdx -= pw;
            if (ijdx >= iw) break;
            int idx = mapping[iidx][ijdx];
            if (idx / slots >= nCT) break;
            idx -= leftshift;
            if (idx < 0) continue;
            for (size_t model_idx = 0; model_idx < nModels; model_idx++)
            {
                auto fdx = idx + offsets[model_idx];
                k[fdx / slots][fdx % slots] = weight[model_idx];
            }
        }
    }
    return k;
}

Tensor<3,Scalar> unmap(const Tensor<3,Scalar>& decoded, const Configuration& config, bool is_kernel, int offset)
{
    const auto& oshape = is_kernel ? config.ishape() : config.oshape();
    const auto& kshape = config.kshape();
    const auto& mapping = is_kernel ? config.imap() : config.omap();
    
    const auto& co = is_kernel ? kshape[1] : oshape[0];
    const auto& oh = oshape[1];
    const auto& ow = oshape[2];
    const auto& slots = Plaintext::default_slots();

    Tensor<3,Scalar> unmapped{co, oh, ow};
    for (size_t o = 0; o < co; o++) // unmapping
    {
        for (size_t bi = 0; bi < oh; bi++)
        {
            for (size_t bj = 0; bj < ow; bj++)
            {
                size_t idx = mapping[bi][bj] + offset;
                unmapped[o][bi][bj] = decoded[o][idx / slots][idx % slots];
            }
        }
    }
    return unmapped;
}

Tensor<4,Scalar> unmap(const Tensor<3,Scalar>& decoded, const Configuration& config, bool is_kernel, const vector<int>& offsets)
{
    Tensor<4,Scalar> unmapped;
    for (int offset : offsets) unmapped.emplace_back(unmap(decoded, config, is_kernel, offset));
    return unmapped;
}

Tensor<4,Scalar> unmap_kernel(const Tensor<6,Scalar>& kernel, const Configuration& config, int offset)
{
    const auto& mapping = config.imap();
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
    const auto& slots = Plaintext::default_slots();

    Tensor<4,Scalar> w{co, ci, kh, kw};
    for (size_t o = 0; o < co; o++) // unmapping
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
                    size_t idx = mapping[iidx][ijdx] + offset;
                    w[o][i][krow][kcol] = kernel[o][i][krow][kcol][idx / slots][idx % slots];
                }
            }
        }
    }
    return w;
}

Tensor<5,Scalar> unmap_kernel(const Tensor<6,Scalar>& kernel, const Configuration& config, const vector<int>& offsets)
{
    Tensor<5,Scalar> unmapped;
    for (int offset : offsets) unmapped.emplace_back(unmap_kernel(kernel, config, offset));
    return unmapped;
}

} // hw

} // fhe