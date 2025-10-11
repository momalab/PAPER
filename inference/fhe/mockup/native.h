#pragma once

#include <vector>
#include "scalar.h"

namespace fhe
{

struct NativePlaintext
{
    std::vector<Scalar> _data;
    std::vector<double> _qi;
    double _scale;
    
    NativePlaintext() = default;
    NativePlaintext(const NativePlaintext&) = default;
    NativePlaintext(NativePlaintext&&) noexcept = default;
    NativePlaintext(const std::vector<Scalar>&, std::size_t slots, const std::vector<int>& logq, double scale);
    NativePlaintext(const std::vector<Scalar>&, std::size_t slots, const std::vector<double>& moduli, double scale);
    NativePlaintext(std::size_t slots, const std::vector<int>& logq, double scale);
    NativePlaintext(std::size_t slots, const std::vector<double>& moduli, double scale);
    NativePlaintext(const Scalar&, std::size_t slots, const std::vector<int>& logq, double scale);
    NativePlaintext(const Scalar&, std::size_t slots, const std::vector<double>& moduli, double scale);
    ~NativePlaintext() = default;

    explicit operator std::vector<Scalar>() const;
    NativePlaintext & operator =(const NativePlaintext&) = default;
    NativePlaintext & operator =(NativePlaintext&&) noexcept = default;
    NativePlaintext& operator+=(const NativePlaintext&);
    NativePlaintext& operator-=(const NativePlaintext&);
    NativePlaintext& operator*=(const NativePlaintext&);
    NativePlaintext& operator<<=(int);
    NativePlaintext operator+(const NativePlaintext&) const;
    NativePlaintext operator-() const;
    NativePlaintext operator-(const NativePlaintext&) const;
    NativePlaintext operator*(const NativePlaintext&) const;
    NativePlaintext operator<<(int) const;

    bool match(const NativePlaintext&) const;
    void modswitch();
    const std::vector<double>& moduli() const;
    void rescale();
    double scale() const;
    std::size_t slots() const;
    std::size_t towers() const;
};

struct NativeCiphertext
{
    NativePlaintext data;
    
    NativeCiphertext() = default;
    NativeCiphertext(const NativeCiphertext&) = default;
    NativeCiphertext(NativeCiphertext&&) noexcept = default;
    NativeCiphertext(const NativePlaintext&);
    NativeCiphertext(const std::vector<Scalar>&, std::size_t slots, const std::vector<int>& logq, double scale);
    NativeCiphertext(const std::vector<Scalar>&, std::size_t slots, const std::vector<double>& moduli, double scale);
    NativeCiphertext(std::size_t slots, const std::vector<int>& logq, double scale);
    NativeCiphertext(std::size_t slots, const std::vector<double>& moduli, double scale);
    NativeCiphertext(const Scalar&, std::size_t slots, const std::vector<int>& logq, double scale);
    NativeCiphertext(const Scalar&, std::size_t slots, const std::vector<double>& moduli, double scale);
    ~NativeCiphertext() = default;

    explicit operator NativePlaintext() const;
    explicit operator std::vector<Scalar>() const;
    NativeCiphertext & operator =(const NativeCiphertext&) = default;
    NativeCiphertext & operator =(NativeCiphertext&&) noexcept = default;
    NativeCiphertext& operator+=(const NativeCiphertext&);
    NativeCiphertext& operator+=(const NativePlaintext&);
    NativeCiphertext& operator-=(const NativeCiphertext&);
    NativeCiphertext& operator-=(const NativePlaintext&);
    NativeCiphertext& operator*=(const NativeCiphertext&);
    NativeCiphertext& operator*=(const NativePlaintext&);
    NativeCiphertext& operator<<=(int);
    NativeCiphertext operator+(const NativeCiphertext&) const;
    NativeCiphertext operator+(const NativePlaintext&) const;
    NativeCiphertext operator-() const;
    NativeCiphertext operator-(const NativeCiphertext&) const;
    NativeCiphertext operator-(const NativePlaintext&) const;
    NativeCiphertext operator*(const NativeCiphertext&) const;
    NativeCiphertext operator*(const NativePlaintext&) const;
    NativeCiphertext operator<<(int) const;

    bool match(const NativeCiphertext&) const;
    bool match(const NativePlaintext&) const;
    void modswitch();
    const std::vector<double>& moduli() const;
    void rescale();
    double scale() const;
    std::size_t slots() const;
    std::size_t towers() const;

};

} // fhe