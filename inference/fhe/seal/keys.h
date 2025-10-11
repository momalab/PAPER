#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "default.h"
#include "native.h"

namespace fhe
{

class Keys
{
    private:
        std::size_t n; // polynomial modulus degree
        std::vector<int> logq; // log2 of ciphertext moduli
        std::vector<seal::Modulus> moduli; // ciphertext moduli
        std::vector<int> steps; // rotation steps
        double _scale; // scaling factor
        seal::EncryptionParameters params;
        std::shared_ptr<seal::SEALContext> context;
        seal::SecretKey sk;
        seal::PublicKey pk;
        seal::RelinKeys rk;
        seal::GaloisKeys gk;
        std::shared_ptr<seal::Encryptor> enc;
        std::shared_ptr<seal::Decryptor> dec;
        std::shared_ptr<seal::Evaluator> eval;
        std::shared_ptr<seal::CKKSEncoder> encoder;
        std::map<int,int> rotation_counter;
        std::vector<NativeCiphertext> ct_zeros;
        std::vector<NativePlaintext> pt_ones; // for rescaling
        std::vector<seal::parms_id_type> parms_ids;
        
        void build_constants();
        void create_context();
        void generate();
        std::pair<const NativeCiphertext*, const NativeCiphertext*> level(const NativeCiphertext&, const NativeCiphertext&, bool=false);
        std::pair<const NativeCiphertext*, const NativePlaintext*> level(const NativeCiphertext&, const NativePlaintext&, bool=false);
        const NativeCiphertext* level_inplace(NativeCiphertext&, const NativeCiphertext&, bool=false);
        const NativePlaintext* level_inplace(NativeCiphertext&, const NativePlaintext&, bool=false);
        void level_inplace(std::vector<NativeCiphertext>&);
        void level_inplace(std::vector<NativeCiphertext*>&);
        void set_scale(double&, double, bool);

        const double SCALE_RATIO_LIMIT = 10; // limit for scaling factor difference

    public:
        Keys() = default;
        Keys(const Keys&) = default;
        Keys(Keys&&) noexcept = default;
        Keys(std::size_t n);
        Keys(std::size_t n, const std::vector<int>& logq, double scale);
        Keys(std::size_t n, const std::vector<int>& logq, double scale, const std::vector<int>& rotation_steps);
        Keys(std::size_t n, const std::vector<int>& logq, double scale, const seal::SecretKey& sk,
            const seal::PublicKey& pk, const seal::RelinKeys& rk, const seal::GaloisKeys& gk);
        ~Keys() = default;

        Keys& operator =(const Keys&) = default;
        Keys& operator =(Keys&&) noexcept = default;
        NativeCiphertext add(const NativeCiphertext&, const NativeCiphertext&);
        NativeCiphertext add(const NativeCiphertext&, const NativePlaintext&);
        NativeCiphertext add(const NativeCiphertext&, const Scalar&);
        NativeCiphertext add(const std::vector<NativeCiphertext>&);
        void add_inplace(NativeCiphertext&, const NativeCiphertext&);
        void add_inplace(NativeCiphertext&, const NativePlaintext&);
        void add_inplace(NativeCiphertext&, const Scalar&);
        void add_inplace(std::vector<NativeCiphertext>&);
        NativeCiphertext* add_inplace(std::vector<NativeCiphertext*>&);
        std::vector<Scalar> decode(const NativeCiphertext&);
        std::vector<Scalar> decode(const NativePlaintext&);
        NativePlaintext decrypt(const NativeCiphertext&);
        NativePlaintext encode(const Scalar&, const NativeCiphertext&);
        NativePlaintext encode(const Scalar&, int = -1, double = 0.0);
        NativePlaintext encode(const Scalar&, const seal::parms_id_type&, double);
        NativePlaintext encode(const std::vector<Scalar>&, const NativeCiphertext&);
        NativePlaintext encode(const std::vector<Scalar>&, int = -1, double = 0.0);
        NativePlaintext encode(const std::vector<Scalar>&, const seal::parms_id_type&, double);
        NativeCiphertext encrypt(const NativePlaintext&);
        NativeCiphertext encrypt(const std::vector<Scalar>&);
        NativeCiphertext encrypt(const Scalar&);
        int level(const NativeCiphertext&) const;
        int level(const NativePlaintext&) const;
        bool load(const std::string& filename);
        NativeCiphertext modswitch(const NativeCiphertext&);
        NativePlaintext modswitch(const NativePlaintext&);
        NativeCiphertext modswitch(const NativeCiphertext&, const NativeCiphertext&);
        NativeCiphertext modswitch(const NativeCiphertext&, const NativePlaintext&);
        NativePlaintext modswitch(const NativePlaintext&, const NativeCiphertext&);
        NativePlaintext modswitch(const NativePlaintext&, const NativePlaintext&);
        void modswitch_inplace(NativeCiphertext&);
        void modswitch_inplace(NativePlaintext&);
        void modswitch_inplace(NativeCiphertext&, const NativeCiphertext&);
        void modswitch_inplace(NativeCiphertext&, const NativePlaintext&);
        void modswitch_inplace(NativePlaintext&, const NativeCiphertext&);
        void modswitch_inplace(NativePlaintext&, const NativePlaintext&);
        NativeCiphertext mul(const NativeCiphertext&, const NativeCiphertext&);
        NativeCiphertext mul(const NativeCiphertext&, const NativePlaintext&);
        NativeCiphertext mul(const NativeCiphertext&, const Scalar&);
        NativeCiphertext mul(const std::vector<NativeCiphertext>&);
        void mul_inplace(NativeCiphertext&, const NativeCiphertext&);
        void mul_inplace(NativeCiphertext&, const NativePlaintext&);
        void mul_inplace(NativeCiphertext&, const Scalar&);
        void mul_inplace(std::vector<NativeCiphertext>&);
        NativeCiphertext* mul_inplace(std::vector<NativeCiphertext*>&);
        NativeCiphertext negate(const NativeCiphertext&);
        void negate_inplace(NativeCiphertext&);
        std::size_t polynomial_degree() const;
        std::uint64_t modulus(int) const;
        NativeCiphertext refit(const NativeCiphertext&);
        void refit_inplace(NativeCiphertext&);
        NativeCiphertext regularize(const NativeCiphertext&);
        void regularize_inplace(NativeCiphertext&);
        NativeCiphertext relinearize(const NativeCiphertext&);
        void relinearize_inplace(NativeCiphertext&);
        NativeCiphertext rescale(const NativeCiphertext&);
        NativeCiphertext rescale(const NativeCiphertext&, const NativeCiphertext&);
        NativeCiphertext rescale(const NativeCiphertext&, const NativePlaintext&);
        void rescale_inplace(NativeCiphertext&);
        void rescale_inplace(NativeCiphertext&, const NativeCiphertext&);
        void rescale_inplace(NativeCiphertext&, const NativePlaintext&);
        NativeCiphertext rotate(const NativeCiphertext&, int);
        void rotate_inplace(NativeCiphertext&, int);
        bool save(const std::string& filename) const;
        double scale() const;
        double scale(const NativeCiphertext&);
        double scale(const NativePlaintext&);
        NativeCiphertext scaleswitch(const NativeCiphertext&);
        NativePlaintext scaleswitch(const NativePlaintext&);
        NativeCiphertext scaleswitch(const NativeCiphertext&, const NativeCiphertext&);
        NativeCiphertext scaleswitch(const NativeCiphertext&, const NativePlaintext&);
        NativePlaintext scaleswitch(const NativePlaintext&, const NativeCiphertext&);
        NativePlaintext scaleswitch(const NativePlaintext&, const NativePlaintext&);
        void scaleswitch_inplace(NativeCiphertext&);
        void scaleswitch_inplace(NativePlaintext&);
        void scaleswitch_inplace(NativeCiphertext&, const NativeCiphertext&);
        void scaleswitch_inplace(NativeCiphertext&, const NativePlaintext&);
        void scaleswitch_inplace(NativeCiphertext&, const Scalar&);
        void scaleswitch_inplace(NativePlaintext&, const NativeCiphertext&);
        void scaleswitch_inplace(NativePlaintext&, const NativePlaintext&);
        // void scaleswitch_inplace(NativePlaintext&, double);
        std::size_t slots() const;
        NativeCiphertext square(const NativeCiphertext&);
        void square_inplace(NativeCiphertext&);
        NativeCiphertext sub(const NativeCiphertext&, const NativeCiphertext&);
        NativeCiphertext sub(const NativeCiphertext&, const NativePlaintext&);
        NativeCiphertext sub(const NativeCiphertext&, const Scalar&);
        void sub_inplace(NativeCiphertext&, const NativeCiphertext&);
        void sub_inplace(NativeCiphertext&, const NativePlaintext&);
        void sub_inplace(NativeCiphertext&, const Scalar&);

        static Keys load_keys(const std::string& filename);
        static bool save_keys(const std::string& filename, const Keys& keys);

        void print_summary() const;
        void print_rotation() const;
        void reset_rotation();
    };

} // fhe
