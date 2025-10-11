#pragma once

#include <vector>
#include "ciphertext.h"
#include "defines.h"
#include "configuration.h"
#include "plaintext.h"
#include "tensor.h"
#include "transform.h"

namespace fhe
{

namespace hw
{


// Decode functions

template <class T> inline
type::Tensor<3,Scalar> decode(const type::Tensor<2,T>& output, const Configuration& config, bool is_kernel, std::size_t nthreads, int offset)
{
    auto bs = transform<3,Scalar,2,T>(output, nthreads); // [decrypting and] decoding
    return unmap(bs, config, is_kernel, offset); // unmapping
}

template <class T> inline
type::Tensor<4,Scalar> decode(const type::Tensor<2,T>& output, const Configuration& config, bool is_kernel, std::size_t nthreads, const std::vector<int>& offsets)
{
    auto bs = transform<3,Scalar,2,T>(output, nthreads); // [decrypting and] decoding
    return unmap(bs, config, is_kernel, offsets); // unmapping
}


// Encode functions using ciphertext as reference

template <class T> inline
type::Tensor<1,T> encode(const type::Tensor<2,Scalar>& input, const Configuration& config, bool is_kernel, std::size_t nthreads, const Ciphertext* reference)
{
    auto mapped = map(input, config, is_kernel); // mapping
    return transform<1,T,2,Scalar>(mapped, nthreads, reference); // encoding [and encrypting]
}

template <class T> inline
type::Tensor<2,T> encode(const type::Tensor<3,Scalar>& input, const Configuration& config, bool is_kernel, std::size_t nthreads, const Ciphertext* reference)
{
    auto mapped = map(input, config, is_kernel); // mapping
    return transform<2,T,3,Scalar>(mapped, nthreads, reference); // encoding [and encrypting]
}

template <class T> inline
type::Tensor<1,T> encode(const type::Tensor<3,Scalar>& input, std::size_t o, const Configuration& config, bool is_kernel, std::size_t nthreads, const Ciphertext* reference)
{
    auto mapped = map(input, o, config, is_kernel); // mapping
    return transform<1,T,2,Scalar>(mapped, nthreads, reference); // encoding [and encrypting]
}

template <class T> inline
type::Tensor<1,T> encode(const type::Tensor<4,Scalar>& input, std::size_t o, std::size_t i, const Configuration& config, bool is_kernel, std::size_t nthreads, const Ciphertext* reference)
{
    auto mapped = map(input, o, i, config, is_kernel); // mapping
    return transform<1,T,2,Scalar>(mapped, nthreads, reference); // encoding [and encrypting]
}

template <class T> inline
type::Tensor<2,T> encode(const type::Tensor<4,Scalar>& input, const Configuration& config, bool is_kernel, std::size_t nthreads, const Ciphertext* reference, const std::vector<int>& offsets)
{
    auto mapped = map(input, config, is_kernel, offsets); // mapping
    return transform<2,T,3,Scalar>(mapped, nthreads, reference); // encoding [and encrypting]
}

template <class T> inline
type::Tensor<1,T> encode(const type::Tensor<4,Scalar>& input, std::size_t o, const Configuration& config, bool is_kernel, std::size_t nthreads, const Ciphertext* reference, const std::vector<int>& offsets)
{
    auto mapped = map(input, o, config, is_kernel, offsets); // mapping
    return transform<1,T,2,Scalar>(mapped, nthreads, reference); // encoding [and encrypting]
}

template <class T> inline type::Tensor<1,T> encode(const type::Tensor<5,Scalar>& kernel, std::size_t o, std::size_t i, const Configuration& config, bool is_kernel, std::size_t nthreads, const Ciphertext* reference, const std::vector<int>& offsets)
{
    auto mapped = map(kernel, o, i, config, is_kernel, offsets); // mapping
    return transform<1,T,2,Scalar>(mapped, nthreads, reference); // encoding [and encrypting]
}


// Encode functions using level and scale

template <class T> inline
type::Tensor<1,T> encode(const type::Tensor<2,Scalar>& input, const Configuration& config, bool is_kernel, std::size_t nthreads, int level, double scale)
{
    auto mapped = map(input, config, is_kernel); // mapping
    return transform<1,T,2,Scalar>(mapped, nthreads, level, scale); // encoding [and encrypting]
}

template <class T> inline
type::Tensor<2,T> encode(const type::Tensor<3,Scalar>& input, const Configuration& config, bool is_kernel, std::size_t nthreads, int level, double scale)
{
    auto mapped = map(input, config, is_kernel); // mapping
    return transform<2,T,3,Scalar>(mapped, nthreads, level, scale); // encoding [and encrypting]
}

template <class T> inline
type::Tensor<1,T> encode(const type::Tensor<3,Scalar>& input, std::size_t o, const Configuration& config, bool is_kernel, std::size_t nthreads, int level, double scale)
{
    auto mapped = map(input, o, config, is_kernel); // mapping
    return transform<1,T,2,Scalar>(mapped, nthreads, level, scale); // encoding [and encrypting]
}

template <class T> inline
type::Tensor<1,T> encode(const type::Tensor<4,Scalar>& input, std::size_t o, std::size_t i, const Configuration& config, bool is_kernel, std::size_t nthreads, int level, double scale)
{
    auto mapped = map(input, o, i, config, is_kernel); // mapping
    return transform<1,T,2,Scalar>(mapped, nthreads, level, scale); // encoding [and encrypting]
}

template <class T> inline
type::Tensor<2,T> encode(const type::Tensor<4,Scalar>& input, const Configuration& config, bool is_kernel, std::size_t nthreads, int level, double scale, const std::vector<int>& offsets)
{
    auto mapped = map(input, config, is_kernel, offsets); // mapping
    return transform<2,T,3,Scalar>(mapped, nthreads, level, scale); // encoding [and encrypting]
}

template <class T> inline
type::Tensor<1,T> encode(const type::Tensor<4,Scalar>& input, std::size_t o, const Configuration& config, bool is_kernel, std::size_t nthreads, int level, double scale, const std::vector<int>& offsets)
{
    auto mapped = map(input, o, config, is_kernel, offsets); // mapping
    return transform<1,T,2,Scalar>(mapped, nthreads, level, scale); // encoding [and encrypting]
}

template <class T> inline type::Tensor<1,T> encode(const type::Tensor<5,Scalar>& kernel, std::size_t o, std::size_t i, const Configuration& config, bool is_kernel, std::size_t nthreads, int level, double scale, const std::vector<int>& offsets)
{
    auto mapped = map(kernel, o, i, config, is_kernel, offsets); // mapping
    return transform<1,T,2,Scalar>(mapped, nthreads, level, scale); // encoding [and encrypting]
}

} // hw

} // fhe