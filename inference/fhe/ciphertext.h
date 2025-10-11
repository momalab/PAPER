#pragma once

#include <cstdint>
#include <iostream>
#include <memory>
#include <tuple>
#include <vector>
#include "keys.h"
#include "plaintext.h"
#include "tensor.h"

namespace fhe
{

class Ciphertext
{
    friend class Plaintext;

    private:
        NativeCiphertext ct;
        std::shared_ptr<Keys> keys;

        static std::shared_ptr<Keys> def_keys;
        
        void pow(int); // exponentiation

    public:
        Ciphertext() = default;
        Ciphertext(const Ciphertext&) = default;
        Ciphertext(Ciphertext&&) noexcept = default;
        Ciphertext(const Plaintext&);
        Ciphertext(const std::vector<Scalar>&);
        Ciphertext(const std::vector<Scalar>&, const std::shared_ptr<Keys>&);
        Ciphertext(const Scalar&);
        Ciphertext(const Scalar&, const std::shared_ptr<Keys>&);
        ~Ciphertext() = default;

        explicit operator Scalar() const;
        explicit operator std::vector<Scalar>() const;
        explicit operator Plaintext() const;

        Ciphertext& operator =(const Ciphertext&) = default;
        Ciphertext& operator =(Ciphertext&&) noexcept = default;

        // compound assignment operators
        Ciphertext& operator ~(); // negate_inplace

        Ciphertext& operator +=(const Ciphertext&);
        Ciphertext& operator *=(const Ciphertext&);
        Ciphertext& operator -=(const Ciphertext&);

        Ciphertext& operator +=(const Plaintext&);
        Ciphertext& operator *=(const Plaintext&);
        Ciphertext& operator -=(const Plaintext&);

        Ciphertext& operator +=(const Scalar&);
        Ciphertext& operator *=(const Scalar&);
        Ciphertext& operator -=(const Scalar&);

        Ciphertext& operator ^=(int); // exponentiation
        Ciphertext& operator <<=(int); // rotate left
        Ciphertext& operator >>=(int); // rotate right

        // const operators
        Ciphertext operator -() const;

        Ciphertext operator +(const Ciphertext&) const;
        Ciphertext operator *(const Ciphertext&) const;
        Ciphertext operator -(const Ciphertext&) const;

        Ciphertext operator +(const Plaintext&) const;
        Ciphertext operator *(const Plaintext&) const;
        Ciphertext operator -(const Plaintext&) const;

        Ciphertext operator +(const Scalar&) const;
        Ciphertext operator *(const Scalar&) const;
        Ciphertext operator -(const Scalar&) const;

        Ciphertext operator ^(int) const; // exponentiation
        Ciphertext operator <<(int) const; // rotate left
        Ciphertext operator >>(int) const; // rotate right

        // friend operators
        friend Ciphertext operator +(const Plaintext&, const Ciphertext&);
        friend Ciphertext operator *(const Plaintext&, const Ciphertext&);
        friend Ciphertext operator -(const Plaintext&, const Ciphertext&);

        friend Ciphertext operator +(const Scalar&, const Ciphertext&);
        friend Ciphertext operator *(const Scalar&, const Ciphertext&);
        friend Ciphertext operator -(const Scalar&, const Ciphertext&);

        // functions
        std::vector<Scalar> decode() const;
        Plaintext decrypt() const;
        double keyscale() const;
        int level() const;
        void modswitch_inplace(int=-1);
        std::size_t polynomial_degree() const;
        std::uint64_t qi() const;
        void refit_inplace();
        void regularize_inplace();
        void relinearize_inplace();
        double scale() const;
        std::size_t slots() const;

        static Ciphertext add(const std::vector<Ciphertext>&, const std::shared_ptr<Keys>& = def_keys);
        static Ciphertext& add_inplace(std::vector<Ciphertext>&, const std::shared_ptr<Keys>& = def_keys);
        static Ciphertext& addslots_inplace(Ciphertext&, std::size_t);
        static const Keys& default_keys(const Keys& keys);
        static const Keys& default_keys(const std::shared_ptr<Keys>& keys = nullptr);
        static std::size_t default_polynomial_degree();
        static std::size_t default_slots();
        static std::tuple<std::vector<std::size_t>,std::vector<std::size_t>> indices_and_lengths(const std::vector<bool>&);
        template <int D> static void modswitch_inplace(type::Tensor<D,Ciphertext>&, int=-1);
        template <int D> static void refit_inplace(type::Tensor<D,Ciphertext>&);
        template <int D> static void regularize_inplace(type::Tensor<D,Ciphertext>&);
        template <int D> static void relinearize_inplace(type::Tensor<D,Ciphertext>&);
        static void shiftleft_inplace(std::vector<Ciphertext>&, int, Scalar = 1.0);
        static void shiftleft_reformat_inplace(std::vector<Ciphertext>&, int, const std::vector<bool>&, const Scalar& = 1.0);

        friend std::ostream& operator <<(std::ostream&, const Ciphertext&);
};

} // fhe

#include "ciphertext.hpp"