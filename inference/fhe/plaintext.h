#pragma once

#include <map>
#include <memory>
#include <tuple>
#include <vector>
#include "keys.h"

namespace fhe
{

class Ciphertext;

class Plaintext
{
    friend class Ciphertext;

    private:
        NativePlaintext pt;
        std::shared_ptr<Keys> keys;

        static std::shared_ptr<Keys> def_keys;

    public:
        Plaintext() = default;
        Plaintext(const Plaintext&) = default;
        Plaintext(Plaintext&&) noexcept = default;
        Plaintext(const std::vector<Scalar>&, const Ciphertext*&);
        Plaintext(const std::vector<Scalar>&, int = -1, double = 0.0);
        Plaintext(const std::vector<Scalar>&, const std::shared_ptr<Keys>&);
        Plaintext(const Scalar&, int = -1, double = 0.0);
        Plaintext(const Scalar&, const Ciphertext*&);
        Plaintext(const Scalar&, const std::shared_ptr<Keys>&);
        ~Plaintext() = default;

        explicit operator Scalar();
        explicit operator std::vector<Scalar>() const;

        Plaintext & operator =(const Plaintext &) = default;
        Plaintext & operator =(Plaintext &&) noexcept = default;

        std::vector<Scalar> decode() const;
        double keyscale() const;
        int level() const;
        size_t polynomial_degree() const;
        double scale() const;
        size_t slots() const;

        static std::map<std::size_t,Plaintext> create_masks(const std::vector<std::size_t> &, const std::vector<std::size_t> &, const Scalar& = 1.0);
        static const Keys& default_keys(const Keys& keys);
        static const Keys& default_keys(const std::shared_ptr<Keys>& keys = nullptr);
        static size_t default_polynomial_degree();
        static size_t default_slots();
        static std::tuple<std::vector<std::size_t>,std::vector<std::size_t>> indices_and_lengths(const std::vector<bool> &);

        friend std::ostream & operator <<(std::ostream & os, const Plaintext & pt);
};

} // fhe