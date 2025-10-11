#pragma once

#include <algorithm>
#include <cstddef>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>
#include "tensor.h"

namespace fhe
{

template <int Dr, class Tr, int Di, class Ti> inline
type::Tensor<Dr,Tr> transform(const type::Tensor<Di,Ti>& t, std::size_t nthreads, const Ciphertext* reference)
{
    static_assert(Dr > 0 && Di > 0, "Dimensions must be positive during transformation");

    std::mutex mtx;
    std::exception_ptr exception;
    std::vector<std::thread> threads(nthreads);
    std::size_t threads_t = std::max(nthreads / t.size(), 1UL);
    if constexpr (Dr == Di) // Same dimensions -- direct mapping
    {
        type::Tensor<Dr,Tr> r(t.shape());
        for (size_t thr = 0; thr < nthreads; thr++) threads[thr] = std::thread([&, thr]()
        {
            try
            {
                if constexpr (Dr == 1) for (size_t i = thr; i < t.size(); i += nthreads) r[i] = Tr(t[i]);
                else for (size_t i = thr; i < t.size(); i += nthreads) r[i] = std::move(transform<Dr-1,Tr,Di-1,Ti>(t[i], threads_t, reference));
            }
            catch (...)
            {
                std::lock_guard<std::mutex> lock(mtx);
                if (!exception) exception = std::current_exception();
            }
        });
        for (auto& thread : threads) thread.join();
        if (exception) std::rethrow_exception(exception);
        return r;
    }
    else if constexpr (Dr > Di) // Decode
    {
        auto rshape = t.shape();
        rshape.push_back(Plaintext::default_slots());
        type::Tensor<Dr,Tr> r(rshape);
        for (size_t thr = 0; thr < nthreads; thr++) threads[thr] = std::thread([&, thr]()
        {
            try
            {
                if constexpr (Di == 1) for (size_t i = thr; i < t.size(); i += nthreads) r[i] = std::move(type::Tensor<Dr-1,Tr>::move(std::vector<Tr>(t[i])));
                else for (size_t i = thr; i < t.size(); i += nthreads) r[i] = std::move(transform<Dr-1,Tr,Di-1,Ti>(t[i], threads_t, reference));
            }
            catch (...)
            {
                std::lock_guard<std::mutex> lock(mtx);
                if (!exception) exception = std::current_exception();
            }
        });
        for (auto& thread : threads) thread.join();
        if (exception) std::rethrow_exception(exception);
        return r;
    }
    else // Encode
    {
        auto rshape = t.shape();
        rshape.pop_back();
        type::Tensor<Dr,Tr> r(rshape);
        for (size_t thr = 0; thr < nthreads; thr++) threads[thr] = std::thread([&, thr]()
        {
            try
            {
                if constexpr (Dr == 1) for (size_t i = thr; i < t.size(); i += nthreads)
                {
                    if constexpr (std::is_same_v<Tr, Plaintext>) r[i] = Tr(t[i].vector(), reference);
                    else r[i] = Tr(t[i].vector());
                }
                else for (size_t i = thr; i < t.size(); i += nthreads) r[i] = std::move(transform<Dr-1,Tr,Di-1,Ti>(t[i], threads_t, reference));
            }
            catch (...)
            {
                std::lock_guard<std::mutex> lock(mtx);
                if (!exception) exception = std::current_exception();
            }
        });
        for (auto& thread : threads) thread.join();
        if (exception) std::rethrow_exception(exception);
        return r;
    }
}

template <int Dr, class Tr, int Di, class Ti> inline
type::Tensor<Dr,Tr> transform(const type::Tensor<Di,Ti>& t, std::size_t nthreads, int level, double scale)
{
    static_assert(Dr > 0 && Di > 0, "Dimensions must be positive during transformation");

    std::mutex mtx;
    std::exception_ptr exception;
    std::vector<std::thread> threads(nthreads);
    std::size_t threads_t = std::max(nthreads / t.size(), 1UL);
    if constexpr (Dr == Di) // Same dimensions -- direct mapping
    {
        type::Tensor<Dr,Tr> r(t.shape());
        for (size_t thr = 0; thr < nthreads; thr++) threads[thr] = std::thread([&, thr]()
        {
            try
            {
                if constexpr (Dr == 1) for (size_t i = thr; i < t.size(); i += nthreads) r[i] = Tr(t[i]);
                else for (size_t i = thr; i < t.size(); i += nthreads) r[i] = std::move(transform<Dr-1,Tr,Di-1,Ti>(t[i], threads_t, level, scale));
            }
            catch (...)
            {
                std::lock_guard<std::mutex> lock(mtx);
                if (!exception) exception = std::current_exception();
            }
        });
        for (auto& thread : threads) thread.join();
        if (exception) std::rethrow_exception(exception);
        return r;
    }
    else if constexpr (Dr > Di) // Decode
    {
        auto rshape = t.shape();
        rshape.push_back(Plaintext::default_slots());
        type::Tensor<Dr,Tr> r(rshape);
        for (size_t thr = 0; thr < nthreads; thr++) threads[thr] = std::thread([&, thr]()
        {
            try
            {
                if constexpr (Di == 1) for (size_t i = thr; i < t.size(); i += nthreads) r[i] = std::move(type::Tensor<Dr-1,Tr>::move(std::vector<Tr>(t[i])));
                else for (size_t i = thr; i < t.size(); i += nthreads) r[i] = std::move(transform<Dr-1,Tr,Di-1,Ti>(t[i], threads_t, level, scale));
            }
            catch (...)
            {
                std::lock_guard<std::mutex> lock(mtx);
                if (!exception) exception = std::current_exception();
            }
        });
        for (auto& thread : threads) thread.join();
        if (exception) std::rethrow_exception(exception);
        return r;
    }
    else // Encode
    {
        auto rshape = t.shape();
        rshape.pop_back();
        type::Tensor<Dr,Tr> r(rshape);
        for (size_t thr = 0; thr < nthreads; thr++) threads[thr] = std::thread([&, thr]()
        {
            try
            {
                if constexpr (Dr == 1) for (size_t i = thr; i < t.size(); i += nthreads)
                {
                    if constexpr (std::is_same_v<Tr, Plaintext>) r[i] = Tr(t[i].vector(), level, scale);
                    else r[i] = Tr(t[i].vector());
                }
                else for (size_t i = thr; i < t.size(); i += nthreads) r[i] = std::move(transform<Dr-1,Tr,Di-1,Ti>(t[i], threads_t, level, scale));
            }
            catch (...)
            {
                std::lock_guard<std::mutex> lock(mtx);
                if (!exception) exception = std::current_exception();
            }
        });
        for (auto& thread : threads) thread.join();
        if (exception) std::rethrow_exception(exception);
        return r;
    }
}

} // fhe