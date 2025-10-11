#pragma once

#include <array>
#include <unordered_map>
#include <utility>
#include <vector>
#include "ciphertext.h"
#include "configuration.h"
#include "defines.h"
#include "hash.h"
#include "plaintext.h"
#include "tensor.h"

namespace fhe
{

namespace hw
{

using KernelMap = std::unordered_map<Scalar, std::pair<type::Tensor<1,Plaintext>, type::Tensor<2,std::array<std::size_t,2>>>>;
using KernelEnsembleMap = std::unordered_map<std::vector<Scalar>, std::pair<type::Tensor<1,Plaintext>, type::Tensor<2,std::array<std::size_t,2>>>, util::VectorDoubleHash, util::VectorDoubleEqual>;

// Mapping functions

type::Tensor<2,Scalar> map(const type::Tensor<2,Scalar>& input, const Configuration& config, bool is_kernel);
type::Tensor<3,Scalar> map(const type::Tensor<3,Scalar>& input, const Configuration& config, bool is_kernel);
type::Tensor<2,Scalar> map(const type::Tensor<3,Scalar>& input, std::size_t o, const Configuration& config, bool is_kernel);
type::Tensor<2,Scalar> map(const type::Tensor<4,Scalar>& input, std::size_t o, std::size_t i, const Configuration& config, bool is_kernel);

type::Tensor<3,Scalar> map(const type::Tensor<4,Scalar>& input, const Configuration& config, bool is_kernel, const std::vector<int>& offsets);
type::Tensor<2,Scalar> map(const type::Tensor<4,Scalar>& input, std::size_t o, const Configuration& config, bool is_kernel, const std::vector<int>& offsets);
type::Tensor<2,Scalar> map(const type::Tensor<5,Scalar>& kernel, std::size_t o, std::size_t i, const Configuration& config, bool is_kernel, const std::vector<int>& offsets);

type::Tensor<3,Scalar> unmap(const type::Tensor<3,Scalar>& decoded, const Configuration& config, bool is_kernel, int offset = 0);
type::Tensor<4,Scalar> unmap(const type::Tensor<3,Scalar>& decoded, const Configuration& config, bool is_kernel, const std::vector<int>& offsets);

type::Tensor<6,Scalar> map_kernel(const type::Tensor<4,Scalar>& kernel, const Configuration& config);
type::Tensor<6,Scalar> map_kernel(const type::Tensor<6,Scalar>& kernel, const Configuration& config, std::size_t nthreads);
type::Tensor<2,Scalar> map_weight(const Scalar& weight, size_t krow, size_t kcol, int leftshift, const Configuration& config);
type::Tensor<2,Scalar> map_weight(const type::Tensor<2,Scalar>& weight, size_t krow, size_t kcol, int leftshift, const Configuration& config);
type::Tensor<2,Scalar> map_weight(const std::vector<Scalar>& weight, size_t krow, size_t kcol, int leftshift, const Configuration& config, const std::vector<int>& offsets);
type::Tensor<4,Scalar> unmap_kernel(const type::Tensor<6,Scalar>& kernel, const Configuration& config, int offset = 0);
type::Tensor<5,Scalar> unmap_kernel(const type::Tensor<6,Scalar>& kernel, const Configuration& config, const std::vector<int>& offsets);


// Template encode/decode functions

template <class T> inline type::Tensor<3,Scalar> decode(const type::Tensor<2,T>& output, const Configuration& config, bool is_kernel, std::size_t nthreads, int offset = 0);
template <class T> inline type::Tensor<4,Scalar> decode(const type::Tensor<2,T>& output, const Configuration& config, bool is_kernel, std::size_t nthreads, const std::vector<int>& offsets);

template <class T> inline type::Tensor<1,T> encode(const type::Tensor<2,Scalar>& bias, const Configuration& config, bool is_kernel, std::size_t nthreads, const Ciphertext* reference);
template <class T> inline type::Tensor<2,T> encode(const type::Tensor<3,Scalar>& bias, const Configuration& config, bool is_kernel, std::size_t nthreads, const Ciphertext* reference);
template <class T> inline type::Tensor<1,T> encode(const type::Tensor<3,Scalar>& bias, std::size_t o, const Configuration& config, bool is_kernel, std::size_t nthreads, const Ciphertext* reference);
template <class T> inline type::Tensor<1,T> encode(const type::Tensor<4,Scalar>& bias, std::size_t o, std::size_t i, const Configuration& config, bool is_kernel, std::size_t nthreads, const Ciphertext* reference);

template <class T> inline type::Tensor<2,T> encode(const type::Tensor<4,Scalar>& bias, const Configuration& config, bool is_kernel, std::size_t nthreads, const Ciphertext* reference, const std::vector<int>& offsets);
template <class T> inline type::Tensor<1,T> encode(const type::Tensor<4,Scalar>& bias, std::size_t o, const Configuration& config, bool is_kernel, std::size_t nthreads, const Ciphertext* reference, const std::vector<int>& offsets);
template <class T> inline type::Tensor<1,T> encode(const type::Tensor<5,Scalar>& bias, std::size_t o, std::size_t i, const Configuration& config, bool is_kernel, std::size_t nthreads, const Ciphertext* reference, const std::vector<int>& offsets);

template <class T> inline type::Tensor<1,T> encode(const type::Tensor<2,Scalar>& bias, const Configuration& config, bool is_kernel, std::size_t nthreads = config::NTHREADS, int level = -1, double scale = 0.0);
template <class T> inline type::Tensor<2,T> encode(const type::Tensor<3,Scalar>& bias, const Configuration& config, bool is_kernel, std::size_t nthreads = config::NTHREADS, int level = -1, double scale = 0.0);
template <class T> inline type::Tensor<1,T> encode(const type::Tensor<3,Scalar>& bias, std::size_t o, const Configuration& config, bool is_kernel, std::size_t nthreads = config::NTHREADS, int level = -1, double scale = 0.0);
template <class T> inline type::Tensor<1,T> encode(const type::Tensor<4,Scalar>& bias, std::size_t o, std::size_t i, const Configuration& config, bool is_kernel, std::size_t nthreads = config::NTHREADS, int level = -1, double scale = 0.0);

template <class T> inline type::Tensor<2,T> encode(const type::Tensor<4,Scalar>& bias, const Configuration& config, bool is_kernel, std::size_t nthreads, int level, double scale, const std::vector<int>& offsets);
template <class T> inline type::Tensor<1,T> encode(const type::Tensor<4,Scalar>& bias, std::size_t o, const Configuration& config, bool is_kernel, std::size_t nthreads, int level, double scale, const std::vector<int>& offsets);
template <class T> inline type::Tensor<1,T> encode(const type::Tensor<5,Scalar>& kernel, std::size_t o, std::size_t i, const Configuration& config, bool is_kernel, std::size_t nthreads, int level, double scale, const std::vector<int>& offsets);


// Specialized decode functions

inline type::Tensor<3,Scalar> decode_bias(const type::Tensor<2,Plaintext>& bias, const Configuration& config, std::size_t nthreads = config::NTHREADS, int offset = 0) { return decode(bias, config, false, nthreads, offset); }
inline type::Tensor<4,Scalar> decode_bias(const type::Tensor<2,Plaintext>& bias, const Configuration& config, std::size_t nthreads, const std::vector<int>& offsets) { return decode(bias, config, false, nthreads, offsets); }
inline type::Tensor<3,Scalar> decode_input(const type::Tensor<2,Ciphertext>& input, const Configuration& config, std::size_t nthreads = config::NTHREADS, int offset = 0) { return decode(input, config, false, nthreads, offset); }
inline type::Tensor<4,Scalar> decode_input(const type::Tensor<2,Ciphertext>& input, const Configuration& config, std::size_t nthreads, const std::vector<int>& offsets) { return decode(input, config, false, nthreads, offsets); }
inline type::Tensor<3,Scalar> decode_kernel(const type::Tensor<2,Plaintext>& kernel, const Configuration& config, std::size_t nthreads = config::NTHREADS, int offset = 0) { return decode(kernel, config, true, nthreads, offset); }
inline type::Tensor<4,Scalar> decode_kernel(const type::Tensor<2,Plaintext>& kernel, const Configuration& config, std::size_t nthreads, const std::vector<int>& offsets) { return decode(kernel, config, true, nthreads, offsets); }
type::Tensor<4,Scalar> decode_kernel(const type::Tensor<5,Plaintext>& kernel, const Configuration& config, std::size_t nthreads = config::NTHREADS, int offset = 0);
type::Tensor<5,Scalar> decode_kernel(const type::Tensor<5,Plaintext>& kernel, const Configuration& config, std::size_t nthreads, const std::vector<int>& offsets);
inline type::Tensor<3,Scalar> decode_output(const type::Tensor<2,Ciphertext>& output, const Configuration& config, std::size_t nthreads = config::NTHREADS, int offset = 0) { return decode(output, config, false, nthreads, offset); }
inline type::Tensor<4,Scalar> decode_output(const type::Tensor<2,Ciphertext>& output, const Configuration& config, std::size_t nthreads, const std::vector<int>& offsets) { return decode(output, config, false, nthreads, offsets); }


// Specialized encode functions using Ciphertext as reference

inline type::Tensor<1,Plaintext> encode_bias(const type::Tensor<2,Scalar>& bias, const Configuration& config, std::size_t nthreads, const Ciphertext* reference) { return encode<Plaintext>(bias, config, false, nthreads, reference); }
inline type::Tensor<2,Plaintext> encode_bias(const type::Tensor<3,Scalar>& bias, const Configuration& config, std::size_t nthreads, const Ciphertext* reference) { return encode<Plaintext>(bias, config, false, nthreads, reference); }
inline type::Tensor<1,Plaintext> encode_bias(const type::Tensor<3,Scalar>& bias, std::size_t o, const Configuration& config, std::size_t nthreads, const Ciphertext* reference) { return encode<Plaintext>(bias, o, config, false, nthreads, reference); }
inline type::Tensor<1,Plaintext> encode_bias(const type::Tensor<4,Scalar>& bias, std::size_t o, const Configuration& config, std::size_t nthreads, const Ciphertext* reference, const std::vector<int>& offsets) { return encode<Plaintext>(bias, o, config, false, nthreads, reference, offsets); }

inline type::Tensor<2,Ciphertext> encode_input(const type::Tensor<3,Scalar>& input, const Configuration& config, std::size_t nthreads, const Ciphertext* reference) { return encode<Ciphertext>(input, config, false, nthreads, reference); }
inline type::Tensor<2,Ciphertext> encode_input(const type::Tensor<4,Scalar>& input, const Configuration& config, std::size_t nthreads, const Ciphertext* reference, const std::vector<int>& offsets) { return encode<Ciphertext>(input, config, false, nthreads, reference, offsets); }

inline type::Tensor<2,Plaintext> encode_kernel(const type::Tensor<3,Scalar>& kernel, const Configuration& config, std::size_t nthreads, const Ciphertext* reference) { return encode<Plaintext>(kernel, config, true, nthreads, reference); }
inline type::Tensor<1,Plaintext> encode_kernel(const type::Tensor<2,Scalar>& kernel, const Configuration& config, std::size_t nthreads, const Ciphertext* reference) { return encode<Plaintext>(kernel, config, true, nthreads, reference); }
inline type::Tensor<1,Plaintext> encode_kernel(const type::Tensor<4,Scalar>& kernel, std::size_t o, std::size_t i, const Configuration& config, std::size_t nthreads, const Ciphertext* reference) { return encode<Plaintext>(kernel, o, i, config, true, nthreads, reference); }
inline type::Tensor<1,Plaintext> encode_kernel(const type::Tensor<5,Scalar>& kernel, std::size_t o, std::size_t i, const Configuration& config, std::size_t nthreads, const Ciphertext* reference, const std::vector<int>& offsets) { return encode<Plaintext>(kernel, o, i, config, true, nthreads, reference, offsets); }

type::Tensor<5,Plaintext> encode_kernel(const type::Tensor<4,Scalar>& kernel, const Configuration& config, std::size_t nthreads, const Ciphertext* reference);
type::Tensor<5,Plaintext> encode_kernel(const type::Tensor<6,Scalar>& kernel, const Configuration& config, std::size_t nthreads, const Ciphertext* reference);

inline type::Tensor<2,Ciphertext> encode_output(const type::Tensor<3,Scalar>& output, const Configuration& config, std::size_t nthreads, const Ciphertext* reference) { return encode<Ciphertext>(output, config, false, nthreads, reference); }

type::Tensor<1,Plaintext> encode_weight(const Scalar& weight, size_t krow, size_t kcol, int leftshift, const Configuration& config, std::size_t nthreads, const Ciphertext* reference);
type::Tensor<1,Plaintext> encode_weight(const type::Tensor<2,Scalar>& weight, size_t krow, size_t kcol, int leftshift, const Configuration& config, std::size_t nthreads, const Ciphertext* reference);
type::Tensor<1,Plaintext> encode_weight(const std::vector<Scalar>& weight, size_t krow, size_t kcol, int leftshift, const Configuration& config, std::size_t nthreads, const Ciphertext* reference, const std::vector<int>& offsets);


// Specialized encode functions using level and scale

inline type::Tensor<1,Plaintext> encode_bias(const type::Tensor<2,Scalar>& bias, const Configuration& config, std::size_t nthreads = config::NTHREADS, int level = -1, double scale = 0.0) { return encode<Plaintext>(bias, config, false, nthreads, level, scale); }
inline type::Tensor<2,Plaintext> encode_bias(const type::Tensor<3,Scalar>& bias, const Configuration& config, std::size_t nthreads = config::NTHREADS, int level = -1, double scale = 0.0) { return encode<Plaintext>(bias, config, false, nthreads, level, scale); }
inline type::Tensor<1,Plaintext> encode_bias(const type::Tensor<3,Scalar>& bias, std::size_t o, const Configuration& config, std::size_t nthreads = config::NTHREADS, int level = -1, double scale = 0.0) { return encode<Plaintext>(bias, o, config, false, nthreads, level, scale); }
inline type::Tensor<1,Plaintext> encode_bias(const type::Tensor<4,Scalar>& bias, std::size_t o, const Configuration& config, std::size_t nthreads, int level, double scale, const std::vector<int>& offsets) { return encode<Plaintext>(bias, o, config, false, nthreads, level, scale, offsets); }

inline type::Tensor<2,Ciphertext> encode_input(const type::Tensor<3,Scalar>& input, const Configuration& config, std::size_t nthreads = config::NTHREADS, int level = -1, double scale = 0.0) { return encode<Ciphertext>(input, config, false, nthreads, level, scale); }
inline type::Tensor<2,Ciphertext> encode_input(const type::Tensor<4,Scalar>& input, const Configuration& config, std::size_t nthreads, int level, double scale, const std::vector<int>& offsets) { return encode<Ciphertext>(input, config, false, nthreads, level, scale, offsets); }

inline type::Tensor<2,Plaintext> encode_kernel(const type::Tensor<3,Scalar>& kernel, const Configuration& config, std::size_t nthreads = config::NTHREADS, int level = -1, double scale = 0.0) { return encode<Plaintext>(kernel, config, true, nthreads, level, scale); }
inline type::Tensor<1,Plaintext> encode_kernel(const type::Tensor<2,Scalar>& kernel, const Configuration& config, std::size_t nthreads = config::NTHREADS, int level = -1, double scale = 0.0) { return encode<Plaintext>(kernel, config, true, nthreads, level, scale); }
inline type::Tensor<1,Plaintext> encode_kernel(const type::Tensor<4,Scalar>& kernel, std::size_t o, std::size_t i, const Configuration& config, std::size_t nthreads = config::NTHREADS, int level = -1, double scale = 0.0) { return encode<Plaintext>(kernel, o, i, config, true, nthreads, level, scale); }
inline type::Tensor<1,Plaintext> encode_kernel(const type::Tensor<5,Scalar>& kernel, std::size_t o, std::size_t i, const Configuration& config, std::size_t nthreads, int level, double scale, const std::vector<int>& offsets) { return encode<Plaintext>(kernel, o, i, config, true, nthreads, level, scale, offsets); }

type::Tensor<5,Plaintext> encode_kernel(const type::Tensor<4,Scalar>& kernel, const Configuration& config, std::size_t nthreads = config::NTHREADS, int level = -1, double scale = 0.0);
type::Tensor<5,Plaintext> encode_kernel(const type::Tensor<6,Scalar>& kernel, const Configuration& config, std::size_t nthreads = config::NTHREADS, int level = -1, double scale = 0.0);

inline type::Tensor<2,Ciphertext> encode_output(const type::Tensor<3,Scalar>& output, const Configuration& config, std::size_t nthreads = config::NTHREADS, int level = -1, double scale = 0.0) { return encode<Ciphertext>(output, config, false, nthreads, level, scale); }

type::Tensor<1,Plaintext> encode_weight(const Scalar& weight, size_t krow, size_t kcol, int leftshift, const Configuration& config, std::size_t nthreads = config::NTHREADS, int level = -1, double scale = 0.0);
type::Tensor<1,Plaintext> encode_weight(const type::Tensor<2,Scalar>& weight, size_t krow, size_t kcol, int leftshift, const Configuration& config, std::size_t nthreads = config::NTHREADS, int level = -1, double scale = 0.0);
type::Tensor<1,Plaintext> encode_weight(const std::vector<Scalar>& weight, size_t krow, size_t kcol, int leftshift, const Configuration& config, std::size_t nthreads, int level, double scale, const std::vector<int>& offsets);

std::vector<KernelMap> encode_kernel_map(const type::Tensor<6,Scalar>& kernel, const Configuration& config, std::size_t nthreads = config::NTHREADS, int level = -1, double scale = 0.0);
KernelMap encode_kernel_map(const type::Tensor<6,Scalar>& kernel, const Configuration& config, std::size_t nthreads, int level, double scale, size_t kcol);
std::vector<KernelEnsembleMap> encode_kernel_map(const type::Tensor<7,Scalar>& kernel, const Configuration& config, std::size_t nthreads, int level, double scale, const std::vector<int>& offsets);
KernelEnsembleMap encode_kernel_map(const type::Tensor<7,Scalar>& kernel, const Configuration& config, std::size_t nthreads, int level, double scale, const std::vector<int>& offsets, size_t kcol);

} // hw

} // fhe

#include "encode.hpp"