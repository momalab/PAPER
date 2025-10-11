#pragma once

#include "tensor.h"

namespace fhe
{

template <int D>
void Ciphertext::modswitch_inplace(type::Tensor<D,Ciphertext>& tensor, int lvl)
{
    if constexpr (D == 1) for (auto& e : tensor) e.modswitch_inplace(lvl);
    else if constexpr (D > 1) for (auto& e : tensor) modswitch_inplace(e, lvl);
    else throw "Tensor dimension must be positive";
}

template <int D>
void Ciphertext::refit_inplace(type::Tensor<D,Ciphertext>& tensor)
{
    if constexpr (D == 1) for (auto& e : tensor) e.refit_inplace();
    else if constexpr (D > 1) for (auto& e : tensor) refit_inplace(e);
    else throw "Tensor dimension must be positive";
}

template <int D>
void Ciphertext::regularize_inplace(type::Tensor<D,Ciphertext>& tensor)
{
    if constexpr (D == 1) for (auto& e : tensor) e.regularize_inplace();
    else if constexpr (D > 1) for (auto& e : tensor) regularize_inplace(e);
    else throw "Tensor dimension must be positive";
}

template <int D>
void Ciphertext::relinearize_inplace(type::Tensor<D,Ciphertext>& tensor)
{
    if constexpr (D == 1) for (auto& e : tensor) e.relinearize_inplace();
    else if constexpr (D > 1) for (auto& e : tensor) relinearize_inplace(e);
    else throw "Tensor dimension must be positive";
}

} // fhe