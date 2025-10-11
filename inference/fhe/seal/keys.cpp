#include "keys.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "math.h"
#include "native.h"

using namespace seal;
using namespace std;

#ifdef RNS_REVERSE
constexpr bool IS_RNS_REVERSE = false;
#else
constexpr bool IS_RNS_REVERSE = true;
#endif

const bool SCALESWITCH  = false;

namespace fhe
{

inline void clean_level(const NativeCiphertext& ct, const NativeCiphertext* ct_ptr)
{
    if (&ct != ct_ptr) delete ct_ptr;
}

inline void clean_level(const NativePlaintext& pt, const NativePlaintext* pt_ptr)
{
    if (&pt != pt_ptr) delete pt_ptr;
}

inline void clean_level(const NativeCiphertext& a, const NativeCiphertext& b, const NativeCiphertext* ct1_ptr, const NativeCiphertext* ct2_ptr)
{
    if (&a != ct1_ptr) delete ct1_ptr;
    if (&b != ct2_ptr) delete ct2_ptr;
}

inline void clean_level(const NativeCiphertext& a, const NativePlaintext& b, const NativeCiphertext* ct_ptr, const NativePlaintext* pt_ptr)
{
    if (&a != ct_ptr) delete ct_ptr;
    if (&b != pt_ptr) delete pt_ptr;
}

Keys::Keys(size_t n)
{
    cout << "Configuring encryption scheme ... " << flush;
    this->n = n;
    if (DEFAULT_PARAMS.find(n) == DEFAULT_PARAMS.end()) throw "fhe::Keys::Keys: Invalid polynomial modulus degree";
    tie(logq, _scale) = DEFAULT_PARAMS.at(n); // choose default parameters logq and scale based on n
    generate();
    build_constants();
    cout << "ok" << endl;
    print_summary();
}

Keys::Keys(size_t n, const vector<int>& logq, double scale)
    : Keys(n, logq, scale, vector<int>()) {}

Keys::Keys(size_t n, const vector<int>& logq, double scale, const vector<int>& rotation_steps)
{
    this->n = n;
    this->logq = logq;
    this->_scale = scale;
    this->steps = rotation_steps;
    int slots = n / 2;
    for (auto& step : this->steps) step = ((step % slots) + slots) % slots;
    generate();
    build_constants();
    print_summary();
}

Keys::Keys(size_t n, const vector<int>& logq, double scale, const SecretKey& sk,
    const PublicKey& pk, const RelinKeys& rk, const GaloisKeys& gk)
{
    this->n = n;
    this->logq = logq;
    this->_scale = scale;
    create_context();
    this->sk = sk;
    this->pk = pk;
    this->rk = rk;
    this->gk = gk;
    enc = make_shared<Encryptor>(*context, pk);
    dec = make_shared<Decryptor>(*context, sk);
    eval = make_shared<Evaluator>(*context);
    encoder = make_shared<CKKSEncoder>(*context);
    build_constants();
}

NativeCiphertext Keys::add(const NativeCiphertext& a, const NativeCiphertext& b)
{
    auto [ct1_ptr, ct2_ptr] = level(a, b, true);
    NativeCiphertext c;
    eval->add(*ct1_ptr, *ct2_ptr, c);
    clean_level(a, b, ct1_ptr, ct2_ptr);
    return c;
}

NativeCiphertext Keys::add(const NativeCiphertext& a, const NativePlaintext& b)
{
    auto [ct_ptr, pt_ptr] = level(a, b, true);
    NativeCiphertext c;
    eval->add_plain(*ct_ptr, *pt_ptr, c);
    clean_level(a, b, ct_ptr, pt_ptr);
    return c;
}

NativeCiphertext Keys::add(const NativeCiphertext& a, const Scalar& b)
{
    return add(a, encode(b, a.parms_id(), a.scale()));
}

NativeCiphertext Keys::add(const vector<NativeCiphertext>& cts)
{
    if (cts.empty()) throw "fhe::Keys::add: No ciphertexts to add";
    size_t size = cts.size();
    vector<NativeCiphertext> v;
    for (size_t i = 0; i < size; i += 2)
    {
        if (i + 1 < size) v.emplace_back(add(cts[i], cts[i + 1]));
        else v.emplace_back(cts[i]);
    }
    add_inplace(v);
    return v.front();
}

void Keys::add_inplace(NativeCiphertext& a, const NativeCiphertext& b)
{
    auto ct2_ptr = level_inplace(a, b, true);
    eval->add_inplace(a, *ct2_ptr);
    clean_level(b, ct2_ptr);
}

void Keys::add_inplace(NativeCiphertext& a, const NativePlaintext& b)
{
    auto pt_ptr = level_inplace(a, b, true);
    eval->add_plain_inplace(a, *pt_ptr);
    clean_level(b, pt_ptr);
}

void Keys::add_inplace(NativeCiphertext& a, const Scalar& b)
{
    eval->add_plain_inplace(a, encode(b, a.parms_id(), a.scale()));
}

void Keys::add_inplace(std::vector<NativeCiphertext>& cts)
{
    if (cts.empty()) throw "fhe::Keys::add_inplace: No ciphertexts to add";
    level_inplace(cts);
    size_t size = cts.size();
    for (size_t half = (size >> 1) + (size & 1); size > 1;)
    {
        for (size_t i = half; i < size; i++) add_inplace(cts[i - half], cts[i]);
        size = half;
        half = (half >> 1) + (half & 1);
    }
    cts.resize(1);
}

NativeCiphertext* Keys::add_inplace(std::vector<NativeCiphertext*>& cts)
{
    if (cts.empty()) throw "fhe::Keys::add_inplace: No ciphertexts to add";
    level_inplace(cts);
    size_t size = cts.size();
    for (size_t half = (size >> 1) + (size & 1); size > 1;)
    {
        for (size_t i = half; i < size; i++) add_inplace(*cts[i - half], *cts[i]);
        size = half;
        half = (half >> 1) + (half & 1);
    }
    cts.resize(1);
    return cts.front();
}

void Keys::build_constants()
{
    ct_zeros.clear();
    pt_ones.clear();
    parms_ids.clear();
    NativeCiphertext zero;
    {
        NativePlaintext zero_pt;
        encoder->encode(0.0, _scale, zero_pt);
        enc->encrypt(zero_pt, zero);
    }
    int lvl = level(zero);
    ct_zeros.resize(lvl + 1);
    pt_ones.resize(lvl + 1);
    parms_ids.resize(lvl + 1);
    ct_zeros[lvl] = zero;
    parms_ids[lvl] = ct_zeros[lvl].parms_id();
    pt_ones[lvl] = encode(1.0, lvl, zero.scale());
    while (--lvl)
    {
        eval->mod_switch_to_next(ct_zeros[lvl + 1], ct_zeros[lvl]);
        parms_ids[lvl] = ct_zeros[lvl].parms_id();
        pt_ones[lvl] = encode(1.0, lvl, ct_zeros[lvl].scale());
    }
}

void Keys::create_context()
{
    params = EncryptionParameters(scheme_type::ckks);
#if RNS_SEAL
    this->moduli = CoeffModulus::Create(n, logq);
#else // RNS_FORWARD or RNS_REVERSE
    if constexpr(!IS_RNS_REVERSE) reverse(logq.begin(), logq.end());
    vector<uint64_t> moduli_uint;
    for (const auto& qi : ::util::find_primes_near_power_of_two(::util::floor_log2(int(n)), logq)) moduli_uint.push_back(qi.get_ui());
    if constexpr(!IS_RNS_REVERSE)
    {
        reverse(moduli_uint.begin(), moduli_uint.end());
        reverse(logq.begin(), logq.end());
    }
    this->moduli.clear();
    for (auto& m : moduli_uint) this->moduli.push_back(Modulus(m));
    cout << "Moduli: " << std::hex;
    for (const auto& m : moduli) cout << "0x" << m.value() << ", ";
    cout << std::dec << endl;
#endif
    params.set_poly_modulus_degree(n);
    params.set_coeff_modulus(moduli);
    context = make_shared<SEALContext>(params, true, sec_level_type::none);
}

vector<Scalar> Keys::decode(const NativeCiphertext& ct)
{
    return decode(decrypt(ct));
}

vector<Scalar> Keys::decode(const NativePlaintext& pt)
{
    vector<Scalar> vs;
    encoder->decode(pt, vs);
    return vs;
}

NativePlaintext Keys::decrypt(const NativeCiphertext& ct)
{
    NativePlaintext pt;
    dec->decrypt(ct, pt);
    return pt;
}

NativePlaintext Keys::encode(const Scalar& scalar, const NativeCiphertext& ct)
{
    return encode(vector<Scalar>(slots(), scalar), ct);
}

NativePlaintext Keys::encode(const Scalar& scalar, int level, double scale)
{
    return encode(vector<Scalar>(slots(), scalar), level, scale);
}

NativePlaintext Keys::encode(const Scalar& scalar, const parms_id_type& parms_id, double scale)
{
    return encode(vector<Scalar>(slots(), scalar), parms_id, scale);
}

NativePlaintext Keys::encode(const vector<Scalar>& vs, const NativeCiphertext& ct)
{
    NativePlaintext pt;
    encoder->encode(vs, ct.parms_id(), ct.scale(), pt);
    return pt;
}

NativePlaintext Keys::encode(const vector<Scalar>& vs, int level, double scale)
{
    const auto& params_id = level < 0 ? parms_ids.back() : parms_ids[level];
    if (scale < 0) throw invalid_argument("Keys::encode: scale must be positive");
    if (!scale) scale = _scale; // use default scale or ct_zeros[level].scale()
    NativePlaintext pt;
    encoder->encode(vs, params_id, scale, pt);
    return pt;
}

NativePlaintext Keys::encode(const vector<Scalar>& vs, const parms_id_type& parms_id, double scale)
{
    NativePlaintext pt;
    encoder->encode(vs, parms_id, scale, pt);
    return pt;
}

NativeCiphertext Keys::encrypt(const NativePlaintext& pt)
{
    NativeCiphertext ct;
    enc->encrypt(pt, ct);
    return ct;
}

NativeCiphertext Keys::encrypt(const vector<Scalar>& vs)
{
    return encrypt(encode(vs));
}

NativeCiphertext Keys::encrypt(const Scalar& scalar)
{
    return encrypt(encode(scalar));
}

void Keys::generate()
{
    create_context();
    KeyGenerator keygen(*context);
    sk = keygen.secret_key();
    keygen.create_public_key(pk);
    keygen.create_relin_keys(rk);
    keygen.create_galois_keys(steps, gk);
    enc = make_shared<Encryptor>(*context, pk);
    dec = make_shared<Decryptor>(*context, sk);
    eval = make_shared<Evaluator>(*context);
    encoder = make_shared<CKKSEncoder>(*context);
}

int Keys::level(const NativeCiphertext& a) const
{
    return a.coeff_modulus_size() - 1;
}

int Keys::level(const NativePlaintext& a) const
{
    return a.coeff_count() / n - 1;
}

pair<const NativeCiphertext*, const NativeCiphertext*> Keys::level(const NativeCiphertext& a, const NativeCiphertext& b, bool setscale)
{
    const NativeCiphertext* ct1_ptr = &a;
    const NativeCiphertext* ct2_ptr = &b;
    if (level(a) > level(b) ||
        (level(a) == level(b) && a.scale() != b.scale()))
    {
        auto ptr = new NativeCiphertext();
        if constexpr (SCALESWITCH) *ptr = scaleswitch(a, b);
        else *ptr = modswitch(a, b);
        set_scale(ptr->scale(), b.scale(), setscale);
        ct1_ptr = ptr;
    }
    else if (level(a) < level(b))
    {
        auto ptr = new NativeCiphertext();
        if constexpr (SCALESWITCH) *ptr = scaleswitch(b, a);
        else *ptr = modswitch(b, a);
        set_scale(ptr->scale(), a.scale(), setscale);
        ct2_ptr = ptr;
    }
    return { ct1_ptr, ct2_ptr };
}

pair<const NativeCiphertext*, const NativePlaintext*> Keys::level(const NativeCiphertext& a, const NativePlaintext& b, bool setscale)
{
    const NativeCiphertext* ct_ptr = &a;
    const NativePlaintext* pt_ptr = &b;
    if (level(a) > level(b) ||
        (level(a) == level(b) && a.scale() != b.scale()))
    {
        auto ptr = new NativeCiphertext();
        if constexpr (SCALESWITCH) *ptr = scaleswitch(a, b);
        else *ptr = modswitch(a, b);
        set_scale(ptr->scale(), b.scale(), setscale);
        ct_ptr = ptr;
    }
    else if (level(a) < level(b))
    {
        auto ptr = new NativePlaintext();
        if constexpr (SCALESWITCH) *ptr = scaleswitch(b, a);
        else *ptr = modswitch(b, a);
        set_scale(ptr->scale(), a.scale(), setscale);
        pt_ptr = ptr;
    }
    return { ct_ptr, pt_ptr };
}

const NativeCiphertext* Keys::level_inplace(NativeCiphertext& a, const NativeCiphertext& b, bool setscale)
{
    const NativeCiphertext* ct2_ptr = &b;
    if (level(a) > level(b) ||
        (level(a) == level(b) && a.scale() != b.scale()))
    {
        if constexpr (SCALESWITCH) scaleswitch_inplace(a, b);
        else modswitch_inplace(a, b);
        set_scale(a.scale(), b.scale(), setscale);
    }
    else if (level(a) < level(b))
    {
        auto ptr = new NativeCiphertext();
        if constexpr (SCALESWITCH) *ptr = scaleswitch(b, a);
        else *ptr = modswitch(b, a);
        set_scale(ptr->scale(), a.scale(), setscale);
        ct2_ptr = ptr;
    }
    return ct2_ptr;
}

const NativePlaintext* Keys::level_inplace(NativeCiphertext& a, const NativePlaintext& b, bool setscale)
{
    const NativePlaintext* pt_ptr = &b;
    if (level(a) > level(b) ||
        (level(a) == level(b) && a.scale() != b.scale()))
    {
        if constexpr (SCALESWITCH) scaleswitch_inplace(a, b);
        else modswitch_inplace(a, b);
        set_scale(a.scale(), b.scale(), setscale);
    }
    else if (level(a) < level(b))
    {
        auto ptr = new NativePlaintext();
        if constexpr (SCALESWITCH) *ptr = scaleswitch(b, a);
        else *ptr = modswitch(b, a);
        set_scale(ptr->scale(), a.scale(), setscale);
        pt_ptr = ptr;
    }
    return pt_ptr;
}

void Keys::level_inplace(vector<NativeCiphertext>& cts)
{
    vector<NativeCiphertext*> ct_ptrs;
    for (auto& ct : cts) ct_ptrs.push_back(&ct);
    level_inplace(ct_ptrs);
}

void Keys::level_inplace(vector<NativeCiphertext*>& cts)
{
    if (cts.empty()) return;

    // find lowest level
    size_t idx = 0;
    for (size_t i = 1; i < cts.size(); i++)
    {
        if constexpr (SCALESWITCH){ if (level(*cts[i]) < level(*cts[idx])) idx = i; }
        else { if (cts[i]->coeff_modulus_size() < cts[idx]->coeff_modulus_size()) idx = i; }
    }
    
    // modswitch all ciphertexts to the lowest level
    for (size_t i = 0; i < cts.size(); i++)
    {
        if constexpr (SCALESWITCH) { if (i != idx) scaleswitch_inplace(*cts[i], *cts[idx]); }
        else { if (i != idx) modswitch_inplace(*cts[i], *cts[idx]); }
    }
}

bool Keys::load(const string& filename)
{
    string fname;
    // n, logq, scale
    try
    {
        fname = filename + ".cfg.key";
        ifstream fin(fname);
        if (fin.fail()) throw "Cannot read '" + fname + "'. Context cannot be created.";
        fin >> n; // read n
        { // read logq
            string line;
            getline(fin, line); // read the rest of the line
            getline(fin, line); // read the next line
            stringstream ss(line);
            logq.clear();
            int logqi;
            while (ss >> logqi) logq.push_back(logqi); // populate logq
        }
        fin >> _scale; // read scale
        create_context();
    }
    catch (...) { throw "Cannot read '" + fname + "'. Context cannot be created."; }

    // secret key
    try
    {
        fname = filename + ".sk.key";
        ifstream fin(fname);
        sk.load(*context, fin);
    }
    catch (...) { std::cout << "WARNING: Cannot read the secret key from '" + fname + "'. Decryption will not work for this key.\n"; }

    // public key
    try
    {
        fname = filename + ".pk.key";
        ifstream fin(fname);
        pk.load(*context, fin);
    }
    catch (...) { std::cout << "WARNING: Cannot read the public key from '" + fname + "'. Encryption will not work for this key.\n"; }

    // relinearization keys
    try
    {
        fname = filename + ".rk.key";
        ifstream fin(fname);
        rk.load(*context, fin);
    }
    catch (...) { std::cout << "WARNING: Cannot read the relinearization keys from '" + fname + "'. Multiplication will not work for this key.\n"; }

    // Galois keys
    try
    {
        fname = filename + ".gk.key";
        ifstream fin(fname);
        gk.load(*context, fin);
    }
    catch (...) { std::cout << "WARNING: Cannot read the Galois keys from '" + fname + "'. Rotations will not work for this key.\n"; }

    return true;
}

Keys Keys::load_keys(const string& filename)
{
    Keys keys;
    if (!keys.load(filename)) throw "Failed to load keys from '" + filename + "'";
    return keys;
}

NativeCiphertext Keys::modswitch(const NativeCiphertext& a)
{
    NativeCiphertext c;
    eval->mod_switch_to_next(a, c);
    return c;
}

NativePlaintext Keys::modswitch(const NativePlaintext& a)
{
    NativePlaintext c;
    eval->mod_switch_to_next(a, c);
    return c;
}

NativeCiphertext Keys::modswitch(const NativeCiphertext& a, const NativeCiphertext& b)
{
    if (level(a) > level(b)) cout << "Keys::modswitch(ct,ct): a.level: " << level(a) << " b.level: " << level(b) << endl;
    NativeCiphertext c;
    eval->mod_switch_to(a, b.parms_id(), c);
    return c;
}

NativeCiphertext Keys::modswitch(const NativeCiphertext& a, const NativePlaintext& b)
{
    if (level(a) > level(b)) cout << "Keys::modswitch(ct,pt): a.level: " << level(a) << " b.level: " << level(b) << endl;
    NativeCiphertext c;
    eval->mod_switch_to(a, b.parms_id(), c);
    return c;
}

NativePlaintext Keys::modswitch(const NativePlaintext& a, const NativeCiphertext& b)
{
    if (level(a) > level(b)) cout << "Keys::modswitch(pt,ct): a.level: " << level(a) << " b.level: " << level(b) << endl;
    NativePlaintext c;
    eval->mod_switch_to(a, b.parms_id(), c);
    return c;
}

NativePlaintext Keys::modswitch(const NativePlaintext& a, const NativePlaintext& b)
{
    if (level(a) > level(b)) cout << "Keys::modswitch(pt,pt): a.level: " << level(a) << " b.level: " << level(b) << endl;
    NativePlaintext c;
    eval->mod_switch_to(a, b.parms_id(), c);
    return c;
}

void Keys::modswitch_inplace(NativeCiphertext& a)
{
    eval->mod_switch_to_next_inplace(a);
}

void Keys::modswitch_inplace(NativePlaintext& a)
{
    eval->mod_switch_to_next_inplace(a);
}

void Keys::modswitch_inplace(NativeCiphertext& a, const NativeCiphertext& b)
{
    if (level(a) > level(b))
    {
        cout << "Keys::modswitch_inplace(ct,ct): a.level: " << level(a) << " b.level: " << level(b) << endl;
        eval->mod_switch_to_inplace(a, b.parms_id());
    }
}

void Keys::modswitch_inplace(NativeCiphertext& a, const NativePlaintext& b)
{
    if (level(a) > level(b))
    {
        cout << "Keys::modswitch_inplace(ct,pt): a.level: " << level(a) << " b.level: " << level(b) << endl;
        eval->mod_switch_to_inplace(a, b.parms_id());
    }
}

void Keys::modswitch_inplace(NativePlaintext& a, const NativeCiphertext& b)
{
    if (level(a) > level(b))
    {
        cout << "Keys::modswitch_inplace(pt,ct): a.level: " << level(a) << " b.level: " << level(b) << endl;
        eval->mod_switch_to_inplace(a, b.parms_id());
    }
}

void Keys::modswitch_inplace(NativePlaintext& a, const NativePlaintext& b)
{
    if (level(a) > level(b))
    {
        cout << "Keys::modswitch_inplace(pt,pt): a.level: " << level(a) << " b.level: " << level(b) << endl;
        eval->mod_switch_to_inplace(a, b.parms_id());
    }
}

uint64_t Keys::modulus(int level) const
{
    return moduli[level].value();
}

NativeCiphertext Keys::mul(const NativeCiphertext& a, const NativeCiphertext& b)
{
    auto [ct1_ptr, ct2_ptr] = level(a, b, false);
    NativeCiphertext c;
    eval->multiply(*ct1_ptr, *ct2_ptr, c);
    clean_level(a, b, ct1_ptr, ct2_ptr);
    return c;
}

NativeCiphertext Keys::mul(const NativeCiphertext& a, const NativePlaintext& b)
{
    auto [ct_ptr, pt_ptr] = level(a, b, false);
    NativeCiphertext c;
    eval->multiply_plain(*ct_ptr, *pt_ptr, c);
    clean_level(a, b, ct_ptr, pt_ptr);
    return c;
}

NativeCiphertext Keys::mul(const NativeCiphertext& a, const Scalar& b)
{
    return mul(a, encode(b, a.parms_id(), _scale));
}

NativeCiphertext Keys::mul(const vector<NativeCiphertext>& cts)
{
    if (cts.empty()) throw "fhe::Keys::mul: No ciphertexts to multiply";
    size_t size = cts.size();
    vector<NativeCiphertext> v;
    for (size_t i = 0; i < size; i += 2)
    {
        if (i + 1 < size)
        {
            v.emplace_back(mul(cts[i], cts[i + 1]));
            relinearize_inplace(v.back());
            rescale_inplace(v.back());
        }
        else v.emplace_back(cts[i]);
    }
    mul_inplace(v);
    return v.front();
}

void Keys::mul_inplace(NativeCiphertext& a, const NativeCiphertext& b)
{
    auto ct2_ptr = level_inplace(a, b, false);
    eval->multiply_inplace(a, *ct2_ptr);
    clean_level(b, ct2_ptr);
}

void Keys::mul_inplace(NativeCiphertext& a, const NativePlaintext& b)
{
    auto pt_ptr = level_inplace(a, b, false);
    eval->multiply_plain_inplace(a, *pt_ptr);
    clean_level(b, pt_ptr);
}

void Keys::mul_inplace(NativeCiphertext& a, const Scalar& b)
{
    mul_inplace(a, encode(b, a.parms_id(), _scale));
}

void Keys::mul_inplace(std::vector<NativeCiphertext>& cts)
{
    if (cts.empty()) throw "fhe::Keys::mul_inplace: No ciphertexts to multiply";
    level_inplace(cts);
    size_t size = cts.size();
    for (size_t half = (size >> 1) + (size & 1); size > 1;)
    {
        for (size_t i = half; i < size; i++)
        {
            mul_inplace(cts[i - half], cts[i]);
            relinearize_inplace(cts[i - half]);
            rescale_inplace(cts[i - half]);
        }
        size = half;
        half = (half >> 1) + (half & 1);
    }
    cts.resize(1);
}

NativeCiphertext* Keys::mul_inplace(std::vector<NativeCiphertext*>& cts)
{
    if (cts.empty()) throw "fhe::Keys::mul_inplace: No ciphertexts to multiply";
    level_inplace(cts);
    size_t size = cts.size();
    for (size_t half = (size >> 1) + (size & 1); size > 1;)
    {
        for (size_t i = half; i < size; i++)
        {
            mul_inplace(*cts[i - half], *cts[i]);
            relinearize_inplace(*cts[i - half]);
            rescale_inplace(*cts[i - half]);
        }
        size = half;
        half = (half >> 1) + (half & 1);
    }
    cts.resize(1);
    return cts.front();
}

NativeCiphertext Keys::negate(const NativeCiphertext& ct)
{
    NativeCiphertext c;
    eval->negate(ct, c);
    return c;
}

void Keys::negate_inplace(NativeCiphertext& ct)
{
    eval->negate_inplace(ct);
}

size_t Keys::polynomial_degree() const
{
    return n;
}

void Keys::print_rotation() const
{
    cout << "Rotation counter: ";
    for (const auto& [key, value] : rotation_counter) cout << key << ":" << value << " ";
    cout << endl;
}

void Keys::print_summary() const
{
    cout << "N: " << polynomial_degree() << endl;
    cout << "logQ: ";
    double qsize = 0.0;
    for (const auto& modulus : moduli)
    {
        double logq = log2(double(modulus.value()));
        cout << round(logq) << " ";
        qsize += logq;
    }
    cout << "= " << round(qsize) << " bits" << endl;
    cout << "Scaling factor: " << scale() << endl;
}

NativeCiphertext Keys::refit(const NativeCiphertext& a)
{
    NativeCiphertext r = a;
    refit_inplace(r);
    return r;
}

void Keys::refit_inplace(NativeCiphertext& ct)
{
    // scale and rescale if the ciphertext's sublevel is greater than one
    double logscale_key = log2(_scale);
    double logscale_ct = log2(ct.scale());
    double sublevel = logscale_ct / logscale_key;
    if (round(sublevel) > 1) // check if sublevel requires refitting
    {
        // new sublevel should be 1; for this, we scale to qi_level + 1, then rescale
        auto lvl = level(ct);
        double qi = modulus(lvl);
        double logqi = log2(qi);
        double qi_level = logqi / logscale_key;
        if (round(qi_level + 1 - sublevel) > 0)
        {
            double scaler = qi * _scale / ct.scale();
            auto one = encode(1.0, lvl, scaler);
            mul_inplace(ct, one);
        }
        rescale_inplace(ct);
    }
}

NativeCiphertext Keys::regularize(const NativeCiphertext& a)
{
    NativeCiphertext r = a;
    regularize_inplace(r);
    return r;
}

void Keys::regularize_inplace(NativeCiphertext& ct)
{
    double logscale_key = log2(_scale);
    double logscale_ct = log2(ct.scale());
    double sublevel = logscale_ct / logscale_key;

    auto lvl = level(ct);
    double qi = modulus(lvl);
    double logqi = log2(qi);
    double qi_level = logqi / logscale_key;

    if (round(sublevel - qi_level) > 0) rescale_inplace(ct);
}

NativeCiphertext Keys::relinearize(const NativeCiphertext& a)
{
    NativeCiphertext c;
    eval->relinearize(a, rk, c);
    return c;
}

void Keys::relinearize_inplace(NativeCiphertext& a)
{
    eval->relinearize_inplace(a, rk);
}

NativeCiphertext Keys::rescale(const NativeCiphertext& a)
{
    NativeCiphertext c;
    eval->rescale_to_next(a, c);
    return c;
}

NativeCiphertext Keys::rescale(const NativeCiphertext& a, const NativeCiphertext& b)
{
    NativeCiphertext c;
    eval->rescale_to(a, b.parms_id(), c);
    return c;
}

NativeCiphertext Keys::rescale(const NativeCiphertext& a, const NativePlaintext& b)
{
    NativeCiphertext c;
    eval->rescale_to(a, b.parms_id(), c);
    return c;
}

void Keys::rescale_inplace(NativeCiphertext& a)
{
    eval->rescale_to_next_inplace(a);
}

void Keys::rescale_inplace(NativeCiphertext& a, const NativeCiphertext& b)
{
    eval->rescale_to_inplace(a, b.parms_id());
}

void Keys::rescale_inplace(NativeCiphertext& a, const NativePlaintext& b)
{
    eval->rescale_to_inplace(a, b.parms_id());
}

void Keys::reset_rotation()
{
    rotation_counter.clear();
}

NativeCiphertext Keys::rotate(const NativeCiphertext& ct, int s)
{
    rotation_counter[s]++;
    NativeCiphertext c;
    eval->rotate_vector(ct, s, gk, c);
    return c;
}

void Keys::rotate_inplace(NativeCiphertext& ct, int s)
{
    rotation_counter[s]++;
    eval->rotate_vector_inplace(ct, s, gk);
}

bool Keys::save(const string& filename) const
{
    try
    {
        // n, logq, scale
        {
            string fname = filename + ".cfg.key";
            ofstream fout(fname);
            fout << n << '\n';
            fout << logq[0];
            for (size_t i = 1; i < logq.size(); i++) fout << ' ' << logq[i];
            fout << '\n';
            ios_base::fmtflags flags = fout.flags(); // save current format
            streamsize precision = cout.precision(); // save current precision
            fout << scientific << setprecision(std::numeric_limits<double>::max_digits10);
            fout << _scale << '\n'; // save scale with maximum precision
            fout.flags(flags); // restore format
            fout.precision(precision); // restore precision
        }
        // secret key
        {
            string fname = filename + ".sk.key";
            ofstream fout(fname);
            sk.save(fout);
        }
        // public key
        {
            string fname = filename + ".pk.key";
            ofstream fout(fname);
            pk.save(fout);
        }
        // relinearization keys
        {
            string fname = filename + ".rk.key";
            ofstream fout(fname);
            rk.save(fout);
        }
        // Galois keys
        {
            string fname = filename + ".gk.key";
            ofstream fout(fname);
            gk.save(fout);
        }
    }
    catch (...) { return false; }
    return true;
}

bool Keys::save_keys(const string& filename, const Keys& keys)
{
    return keys.save(filename);
}

double Keys::scale() const
{
    return _scale;
}

double Keys::scale(const NativeCiphertext& ct)
{
    return ct.scale();
}

double Keys::scale(const NativePlaintext& pt)
{
    return pt.scale();
}

NativeCiphertext Keys::scaleswitch(const NativeCiphertext& ct)
{
    NativeCiphertext c;
    eval->multiply_plain(ct, pt_ones.at(level(ct)), c);
    rescale_inplace(c);
    return c;
}

NativePlaintext Keys::scaleswitch(const NativePlaintext& pt)
{
    int new_level = level(pt) - 1;
    return encode(decode(pt), new_level);
}

NativeCiphertext Keys::scaleswitch(const NativeCiphertext& a, const NativeCiphertext& b)
{
    if (level(a) < level(b)) throw string("fhe::Keys::scaleswitch: Target ciphertext has higher level than source ciphertext");
    NativeCiphertext r = a;
    scaleswitch_inplace(r, b);
    return r;
}

NativeCiphertext Keys::scaleswitch(const NativeCiphertext& a, const NativePlaintext& b)
{
    if (level(a) < level(b)) throw string("fhe::Keys::scaleswitch: Target plaintext has higher level than source ciphertext");
    NativeCiphertext r = a;
    scaleswitch_inplace(r, b);
    return r;
}

NativePlaintext Keys::scaleswitch(const NativePlaintext& a, const NativeCiphertext& b)
{
    if (level(a) < level(b)) throw string("fhe::Keys::scaleswitch: Target ciphertext has higher level than source plaintext");
    NativePlaintext r = a;
    scaleswitch_inplace(r, b);
    return r;
}

NativePlaintext Keys::scaleswitch(const NativePlaintext& a, const NativePlaintext& b)
{
    if (level(a) < level(b)) throw string("fhe::Keys::scaleswitch: Target plaintext has higher level than source plaintext");
    NativePlaintext r = a;
    scaleswitch_inplace(r, b);
    return r;
}

void Keys::scaleswitch_inplace(NativeCiphertext& ct)
{
    const auto& pt_one = pt_ones.at(level(ct));
    eval->multiply_plain_inplace(ct, pt_one);
    rescale_inplace(ct);
}

void Keys::scaleswitch_inplace(NativePlaintext& pt)
{
    int new_level = level(pt) - 1;
    pt = encode(decode(pt), new_level);
}

void Keys::scaleswitch_inplace(NativeCiphertext& a, const NativeCiphertext& b)
{
    if (level(a) < level(b)) throw string("fhe::Keys::scaleswitch_inplace: Target ciphertext has higher level than source ciphertext");
    // while (level(a) != level(b)) scaleswitch_inplace(a);
    while (level(a) > level(b) + 1) scaleswitch_inplace(a);
    cout << "M::scaleswitch_inplace(ct,ct): " << a.scale() << " " << b.scale() << endl;
    if (level(a) > level(b)) scaleswitch_inplace(a, b.scale() / a.scale());
    cout << "O::scaleswitch_inplace(ct,ct): " << a.scale() << " " << b.scale() << endl;
}

void Keys::scaleswitch_inplace(NativeCiphertext& a, const NativePlaintext& b)
{
    if (level(a) < level(b)) throw string("fhe::Keys::scaleswitch_inplace: Target plaintext has higher level than source ciphertext");
    // while (level(a) != level(b)) scaleswitch_inplace(a);
    while (level(a) > level(b) + 1) scaleswitch_inplace(a);
    cout << "M::scaleswitch_inplace(ct,pt): " << a.scale() << " " << b.scale() << endl;
    if (level(a) > level(b)) scaleswitch_inplace(a, b.scale() / a.scale());
    cout << "O::scaleswitch_inplace(ct,pt): " << a.scale() << " " << b.scale() << endl;
}

void Keys::scaleswitch_inplace(NativeCiphertext& ct, const Scalar& scaling_factor)
{
    cout << "Scaling factor: " << scaling_factor << endl;
    eval->multiply_plain_inplace(ct, encode(1.0, ct.parms_id(), _scale * scaling_factor));
    rescale_inplace(ct);
}

void Keys::scaleswitch_inplace(NativePlaintext& a, const NativeCiphertext& b)
{
    if (level(a) < level(b)) throw string("fhe::Keys::scaleswitch_inplace: Target ciphertext has higher level than source plaintext");
    while (level(a) != level(b)) scaleswitch_inplace(a);
}

void Keys::scaleswitch_inplace(NativePlaintext& a, const NativePlaintext& b)
{
    if (level(a) < level(b)) throw string("fhe::Keys::scaleswitch_inplace: Target plaintext has higher level than source plaintext");
    while (level(a) != level(b)) scaleswitch_inplace(a);
}

void Keys::set_scale(double& scale1, double scale2, bool set_scale)
{
    if (!set_scale) return;
    auto min_s = min(scale1, scale2);
    auto max_s = max(scale1, scale2);
    if (max_s / min_s > SCALE_RATIO_LIMIT)
    {
        cout << "set_scale::scale mismatch: " << scale1 << " " << scale2 << endl;
        throw string("fhe::Keys::set_scale: Scale mismatch (") + to_string(scale1) + ", " + to_string(scale2) + ")";
    }
    scale1 = scale2;
}

size_t Keys::slots() const
{
    return encoder->slot_count();
}

NativeCiphertext Keys::square(const NativeCiphertext& ct)
{
    NativeCiphertext c;
    eval->square(ct, c);
    return c;
}

void Keys::square_inplace(NativeCiphertext& ct)
{
    eval->square_inplace(ct);
}

NativeCiphertext Keys::sub(const NativeCiphertext& a, const NativeCiphertext& b)
{
    auto [ct1_ptr, ct2_ptr] = level(a, b, true);
    NativeCiphertext c;
    eval->sub(*ct1_ptr, *ct2_ptr, c);
    clean_level(a, b, ct1_ptr, ct2_ptr);
    return c;
}

NativeCiphertext Keys::sub(const NativeCiphertext& a, const NativePlaintext& b)
{
    auto [ct_ptr, pt_ptr] = level(a, b, true);
    NativeCiphertext c;
    eval->sub_plain(*ct_ptr, *pt_ptr, c);
    clean_level(a, b, ct_ptr, pt_ptr);
    return c;
}

NativeCiphertext Keys::sub(const NativeCiphertext& a, const Scalar& b)
{
    return sub(a, encode(b, a.parms_id(), _scale));
}

void Keys::sub_inplace(NativeCiphertext& a, const NativeCiphertext& b)
{
    auto ct2_ptr = level_inplace(a, b, true);
    eval->sub_inplace(a, *ct2_ptr);
    clean_level(b, ct2_ptr);
}

void Keys::sub_inplace(NativeCiphertext& a, const NativePlaintext& b)
{
    auto pt_ptr = level_inplace(a, b, true);
    eval->sub_plain_inplace(a, *pt_ptr);
    clean_level(b, pt_ptr);
}

void Keys::sub_inplace(NativeCiphertext& a, const Scalar& b)
{
    eval->sub_plain_inplace(a, encode(b, a.parms_id(), _scale));
}

} // fhe
