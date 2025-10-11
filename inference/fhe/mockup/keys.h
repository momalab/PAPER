#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include "default.h"
#include "native.h"

namespace fhe
{

class Keys
{
    private:
        std::size_t n; // polynomial modulus degree
        std::vector<int> logq; // log2 of the moduli
        double _scale; // scaling factor
        std::map<int,int> rotation_counter;

        std::pair<const NativeCiphertext*, const NativeCiphertext*> level(const NativeCiphertext&, const NativeCiphertext&, bool=false) const;
        std::pair<const NativeCiphertext*, const NativePlaintext*> level(const NativeCiphertext&, const NativePlaintext&, bool=false) const;
        const NativeCiphertext* level_inplace(NativeCiphertext&, const NativeCiphertext&, bool=false) const;
        const NativePlaintext* level_inplace(NativeCiphertext&, const NativePlaintext&, bool=false) const;
        void level_inplace(std::vector<NativeCiphertext>&) const;
        void level_inplace(std::vector<NativeCiphertext*>&) const;
        
    public:
        Keys() = default;
        Keys(const Keys&) = default;
        Keys(Keys&&) noexcept = default;
        Keys(std::size_t n);
        Keys(std::size_t n, const std::vector<int>& logq, double scale);
        Keys(std::size_t n, const std::vector<int>& logq, double scale, const std::vector<int>& rotation_steps);
        ~Keys() = default;

        Keys& operator =(const Keys&) = default;
        Keys& operator =(Keys&&) noexcept = default;
        NativeCiphertext add(const NativeCiphertext&, const NativeCiphertext&) const;
        NativeCiphertext add(const NativeCiphertext&, const NativePlaintext&) const;
        NativeCiphertext add(const NativeCiphertext&, const Scalar&) const;
        NativeCiphertext add(const std::vector<NativeCiphertext>&) const;
        void add_inplace(NativeCiphertext&, const NativeCiphertext&) const;
        void add_inplace(NativeCiphertext&, const NativePlaintext&) const;
        void add_inplace(NativeCiphertext&, const Scalar&) const;
        void add_inplace(std::vector<NativeCiphertext>&) const;
        NativeCiphertext* add_inplace(std::vector<NativeCiphertext*>&) const;
        std::vector<Scalar> decode(const NativeCiphertext&) const;
        std::vector<Scalar> decode(const NativePlaintext&) const;
        NativePlaintext decrypt(const NativeCiphertext&) const;
        NativePlaintext encode(const Scalar&, const NativeCiphertext&) const;
        NativePlaintext encode(const Scalar&, int = -1, double = 0.0) const;
        NativePlaintext encode(const std::vector<Scalar>&, int = -1, double = 0.0) const;
        NativePlaintext encode(const std::vector<Scalar>&, const NativeCiphertext&) const;
        NativeCiphertext encrypt(const NativePlaintext&) const;
        NativeCiphertext encrypt(const std::vector<Scalar>&) const;
        NativeCiphertext encrypt(const Scalar&) const;
        int level(const NativeCiphertext&) const;
        int level(const NativePlaintext&) const;
        bool load(const std::string& filename);
        NativeCiphertext modswitch(const NativeCiphertext&) const;
        NativePlaintext modswitch(const NativePlaintext&) const;
        NativeCiphertext modswitch(const NativeCiphertext&, const NativeCiphertext&) const;
        NativeCiphertext modswitch(const NativeCiphertext&, const NativePlaintext&) const;
        NativePlaintext modswitch(const NativePlaintext&, const NativeCiphertext&) const;
        NativePlaintext modswitch(const NativePlaintext&, const NativePlaintext&) const;
        void modswitch_inplace(NativeCiphertext&) const;
        void modswitch_inplace(NativePlaintext&) const;
        void modswitch_inplace(NativeCiphertext&, const NativeCiphertext&) const;
        void modswitch_inplace(NativeCiphertext&, const NativePlaintext&) const;
        void modswitch_inplace(NativePlaintext&, const NativeCiphertext&) const;
        void modswitch_inplace(NativePlaintext&, const NativePlaintext&) const;
        std::uint64_t modulus(int) const;
        NativeCiphertext mul(const NativeCiphertext&, const NativeCiphertext&) const;
        NativeCiphertext mul(const NativeCiphertext&, const NativePlaintext&) const;
        NativeCiphertext mul(const NativeCiphertext&, const Scalar&) const;
        NativeCiphertext mul(const std::vector<NativeCiphertext>&) const;
        void mul_inplace(NativeCiphertext&, const NativeCiphertext&) const;
        void mul_inplace(NativeCiphertext&, const NativePlaintext&) const;
        void mul_inplace(NativeCiphertext&, const Scalar&) const;
        void mul_inplace(std::vector<NativeCiphertext>&) const;
        NativeCiphertext* mul_inplace(std::vector<NativeCiphertext*>&) const;
        NativeCiphertext negate(const NativeCiphertext&) const;
        void negate_inplace(NativeCiphertext&) const;
        std::size_t polynomial_degree() const;
        NativeCiphertext refit(const NativeCiphertext&) const;
        void refit_inplace(NativeCiphertext&) const;
        NativeCiphertext regularize(const NativeCiphertext&) const;
        void regularize_inplace(NativeCiphertext&) const;
        NativeCiphertext relinearize(const NativeCiphertext&) const;
        void relinearize_inplace(NativeCiphertext&) const;
        void rescale_inplace(NativeCiphertext&) const;
        NativeCiphertext rotate(const NativeCiphertext&, int);
        void rotate_inplace(NativeCiphertext&, int);
        bool save(const std::string& filename) const;
        double scale() const;
        double scale(const NativeCiphertext&) const;
        double scale(const NativePlaintext&) const;
        std::size_t slots() const;
        NativeCiphertext square(const NativeCiphertext&) const;
        void square_inplace(NativeCiphertext&) const;
        NativeCiphertext sub(const NativeCiphertext&, const NativeCiphertext&) const;
        NativeCiphertext sub(const NativeCiphertext&, const NativePlaintext&) const;
        NativeCiphertext sub(const NativeCiphertext&, const Scalar&) const;
        void sub_inplace(NativeCiphertext&, const NativeCiphertext&) const;
        void sub_inplace(NativeCiphertext&, const NativePlaintext&) const;
        void sub_inplace(NativeCiphertext&, const Scalar&) const;

        static Keys load_keys(const std::string& filename);
        static bool save_keys(const std::string& filename, const Keys& keys);

        void print_rotation() const;
        void reset_rotation();
};

} // fhe