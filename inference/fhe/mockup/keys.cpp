#include "keys.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>
#include "io.h"
#include "native.h"

using namespace std;

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
    this->n = n;
    if (DEFAULT_PARAMS.find(n) == DEFAULT_PARAMS.end()) throw "fhe::Keys::Keys: Invalid polynomial modulus degree";
    tie(logq, _scale) = DEFAULT_PARAMS.at(n); // choose default parameters logq and scale based on n
    cout << "fhe::Keys::Keys: Using default parameters for n = " << n << ": logq = " << logq << ", scale = " << _scale << endl;
}

Keys::Keys(size_t n, const vector<int>& logq, double scale)
{
    if (logq.empty()) throw "fhe::Keys::Keys: logq is empty";
    if (scale < 2.0) throw "fhe::Keys::Keys: scale is too small";
    this->n = n;
    this->logq = logq;
    this->logq.pop_back(); // remove log(P) as it is not needed for mockup
    this->_scale = scale;
    cout << "fhe::Keys::Keys: Using custom parameters for n = " << n << ": logq = " << logq << ", scale = " << scale << endl;
}

Keys::Keys(size_t n, const vector<int>& logq, double scale, const vector<int>& rotation_steps)
    : Keys(n, logq, scale)
{
    (void) rotation_steps;
}

NativeCiphertext Keys::add(const NativeCiphertext& a, const NativeCiphertext& b) const
{
    auto [ct1_ptr, ct2_ptr] = level(a, b, true);
    NativeCiphertext c = *ct1_ptr + *ct2_ptr;
    clean_level(a, b, ct1_ptr, ct2_ptr);
    return c;
}

NativeCiphertext Keys::add(const NativeCiphertext& a, const NativePlaintext& b) const
{
    auto [ct_ptr, pt_ptr] = level(a, b, true);
    NativeCiphertext c = *ct_ptr + *pt_ptr;
    clean_level(a, b, ct_ptr, pt_ptr);
    return c;
}

NativeCiphertext Keys::add(const NativeCiphertext& a, const Scalar& b) const
{
    return add(a, encode(b, a));
}

NativeCiphertext Keys::add(const vector<NativeCiphertext>& cts) const
{
    if (cts.empty()) throw "fhe::Keys::add: No ciphertexts to add";
    NativeCiphertext r = cts.front();
    for (size_t i = 1; i < cts.size(); i++) add_inplace(r, cts[i]);
    return r;
}

void Keys::add_inplace(NativeCiphertext& a, const NativeCiphertext& b) const
{
    auto ct2_ptr = level_inplace(a, b, true);
    a += *ct2_ptr;
    clean_level(b, ct2_ptr);
}

void Keys::add_inplace(NativeCiphertext& a, const NativePlaintext& b) const
{
    auto pt_ptr = level_inplace(a, b, true);
    a += *pt_ptr;
    clean_level(b, pt_ptr);
}

void Keys::add_inplace(NativeCiphertext& a, const Scalar& b) const
{
    add_inplace(a, encode(b, a));
}

void Keys::add_inplace(std::vector<NativeCiphertext>& cts) const
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

NativeCiphertext* Keys::add_inplace(std::vector<NativeCiphertext*>& cts) const
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

vector<Scalar> Keys::decode(const NativeCiphertext& ct) const
{
    return vector<Scalar>(ct);
}

vector<Scalar> Keys::decode(const NativePlaintext& pt) const
{
    return vector<Scalar>(pt);
}

NativePlaintext Keys::decrypt(const NativeCiphertext& ct) const
{
    return (NativePlaintext) ct;
}

NativePlaintext Keys::encode(const Scalar& scalar, const NativeCiphertext& ct) const
{
    return NativePlaintext(scalar, ct.slots(), ct.moduli(), ct.scale());
}

NativePlaintext Keys::encode(const Scalar& scalar, int level, double scale) const
{
    if (scale == 0.0) scale = _scale;
    if (level < 0) level = logq.size() - 1;
    if (level >= int(logq.size())) throw "fhe::Keys::encode: level is too large (" + to_string(level) + ")";
    if (scale < 2.0) throw "fhe::Keys::encode: scale is too small (" + to_string(scale) + ")";
    if (!scale) scale = _scale;
    auto logq_level = logq;
    logq_level.resize(level + 1);
    return NativePlaintext(scalar, slots(), logq_level, scale);
}

NativePlaintext Keys::encode(const vector<Scalar>& vs, const NativeCiphertext& ct) const
{
    NativePlaintext pt(vs, ct.slots(), ct.moduli(), ct.scale());
    return pt;
}

NativePlaintext Keys::encode(const vector<Scalar>& vs, int level, double scale) const
{
    if (scale == 0.0) scale = _scale;
    if (level < 0) level = logq.size() - 1;
    if (level >= int(logq.size())) throw "fhe::Keys::encode: level is too large (" + to_string(level) + ")";
    if (scale < 2.0) throw "fhe::Keys::encode: scale is too small (" + to_string(scale) + ")";
    if (!scale) scale = _scale;
    auto logq_level = logq;
    logq_level.resize(level + 1);
    NativePlaintext pt(vs, slots(), logq_level, scale);
    return pt;
}

NativeCiphertext Keys::encrypt(const NativePlaintext& pt) const
{
    return NativeCiphertext(pt);
}

NativeCiphertext Keys::encrypt(const vector<Scalar>& vs) const
{
    NativeCiphertext ct(vs, slots(), logq, _scale);
    return ct;
}

NativeCiphertext Keys::encrypt(const Scalar& scalar) const
{
    return NativeCiphertext(scalar, slots(), logq, _scale);
}

int Keys::level(const NativeCiphertext& ct) const
{
    return ct.towers() - 1;
}

int Keys::level(const NativePlaintext& pt) const
{
    return pt.towers() - 1;
}

pair<const NativeCiphertext*, const NativeCiphertext*> Keys::level(const NativeCiphertext& a, const NativeCiphertext& b, bool setscale) const
{
    (void) setscale;
    const NativeCiphertext* ct1_ptr = &a;
    const NativeCiphertext* ct2_ptr = &b;
    if (level(a) > level(b)) ct1_ptr = new NativeCiphertext(modswitch(a, b));
    else if (level(a) < level(b)) ct2_ptr = new NativeCiphertext(modswitch(b, a));
    return { ct1_ptr, ct2_ptr };
}

pair<const NativeCiphertext*, const NativePlaintext*> Keys::level(const NativeCiphertext& a, const NativePlaintext& b, bool setscale) const
{
    (void) setscale;
    const NativeCiphertext* ct_ptr = &a;
    const NativePlaintext* pt_ptr = &b;
    if (level(a) > level(b)) ct_ptr = new NativeCiphertext(modswitch(a, b));
    else if (level(a) < level(b)) pt_ptr = new NativePlaintext(modswitch(b, a));
    return { ct_ptr, pt_ptr };
}

const NativeCiphertext* Keys::level_inplace(NativeCiphertext& a, const NativeCiphertext& b, bool setscale) const
{
    (void) setscale;
    const NativeCiphertext* ct2_ptr = &b;
    if (level(a) > level(b)) modswitch_inplace(a, b);
    else if (level(a) < level(b)) ct2_ptr = new NativeCiphertext(modswitch(b, a)); 
    return ct2_ptr;
}

const NativePlaintext* Keys::level_inplace(NativeCiphertext& a, const NativePlaintext& b, bool setscale) const
{
    (void) setscale;
    const NativePlaintext* pt_ptr = &b;
    if (level(a) > level(b)) modswitch_inplace(a, b);
    else if (level(a) < level(b)) pt_ptr = new NativePlaintext(modswitch(b, a));
    return pt_ptr;
}

void Keys::level_inplace(vector<NativeCiphertext>& cts) const
{
    vector<NativeCiphertext*> ct_ptrs;
    for (auto& ct : cts) ct_ptrs.push_back(&ct);
    level_inplace(ct_ptrs);
}

void Keys::level_inplace(vector<NativeCiphertext*>& cts) const
{
    if (cts.empty()) return;

    // find lowest level
    size_t idx = 0;
    for (size_t i = 1; i < cts.size(); i++) if (cts[i]->towers() < cts[idx]->towers()) idx = i;
    
    // modswitch all ciphertexts to the lowest level
    for (size_t i = 0; i < cts.size(); i++) if (i != idx) modswitch_inplace(*cts[i], *cts[idx]);
}

bool Keys::load(const string& filename)
{
    ifstream fin(filename);
    if (fin.fail()) return false;
    fin >> n;
    return true;
}

Keys Keys::load_keys(const string& filename)
{
    Keys keys;
    if (!keys.load(filename)) throw "Failed to load keys from '" + filename + "'";
    return keys;
}

NativeCiphertext Keys::modswitch(const NativeCiphertext& a) const
{
    NativeCiphertext c = a;
    modswitch_inplace(c);
    return c;
}

NativePlaintext Keys::modswitch(const NativePlaintext& a) const
{
    NativePlaintext c = a;
    modswitch_inplace(c);
    return c;
}

NativeCiphertext Keys::modswitch(const NativeCiphertext& a, const NativeCiphertext& b) const
{
    NativeCiphertext c = a;
    modswitch_inplace(c, b);
    return c;
}

NativeCiphertext Keys::modswitch(const NativeCiphertext& a, const NativePlaintext& b) const
{
    NativeCiphertext c = a;
    modswitch_inplace(c, b);
    return c;
}

NativePlaintext Keys::modswitch(const NativePlaintext& a, const NativeCiphertext& b) const
{
    NativePlaintext c = a;
    modswitch_inplace(c, b);
    return c;
}

NativePlaintext Keys::modswitch(const NativePlaintext& a, const NativePlaintext& b) const
{
    NativePlaintext c = a;
    modswitch_inplace(c, b);
    return c;
}

uint64_t Keys::modulus(int level) const
{
    if (level < 0 || level >= int(logq.size())) throw "fhe::Keys::modulus: Invalid level " + to_string(level);
    return (1ULL << logq[level]);
}

void Keys::modswitch_inplace(NativeCiphertext& a) const
{
    a.modswitch();
}

void Keys::modswitch_inplace(NativePlaintext& a) const
{
    a.modswitch();
}

void Keys::modswitch_inplace(NativeCiphertext& a, const NativeCiphertext& b) const
{
    if (level(a) < level(b)) throw "fhe::Keys::modswitch_inplace: Target ciphertext has lower level than source ciphertext";
    while (level(a) > level(b)) modswitch_inplace(a);
}

void Keys::modswitch_inplace(NativeCiphertext& a, const NativePlaintext& b) const
{
    if (level(a) < level(b)) throw "fhe::Keys::modswitch_inplace: Target plaintext has lower level than source ciphertext";
    while (level(a) > level(b)) modswitch_inplace(a);
}

void Keys::modswitch_inplace(NativePlaintext& a, const NativeCiphertext& b) const
{
    if (level(a) < level(b)) throw "fhe::Keys::modswitch_inplace: Target ciphertext has lower level than source plaintext";
    while (level(a) > level(b)) modswitch_inplace(a);
}

void Keys::modswitch_inplace(NativePlaintext& a, const NativePlaintext& b) const
{
    if (level(a) < level(b)) throw "fhe::Keys::modswitch_inplace: Target plaintext has lower level than source plaintext";
    while (level(a) > level(b)) modswitch_inplace(a);
}

NativeCiphertext Keys::mul(const NativeCiphertext& a, const NativeCiphertext& b) const
{
    auto [ct1_ptr, ct2_ptr] = level(a, b, false);
    NativeCiphertext c = *ct1_ptr * *ct2_ptr;
    clean_level(a, b, ct1_ptr, ct2_ptr);
    return c;
}

NativeCiphertext Keys::mul(const NativeCiphertext& a, const NativePlaintext& b) const
{
    auto [ct_ptr, pt_ptr] = level(a, b, false);
    NativeCiphertext c = *ct_ptr * *pt_ptr;
    clean_level(a, b, ct_ptr, pt_ptr);
    return c;
}

NativeCiphertext Keys::mul(const NativeCiphertext& a, const Scalar& b) const
{
    return mul(a, encode(b, a.towers() - 1, _scale)); // default scale
}

NativeCiphertext Keys::mul(const vector<NativeCiphertext>& cts) const
{
    if (cts.empty()) throw "fhe::Keys::mul: No ciphertexts to multiply";
    size_t size = cts.size();
    vector<NativeCiphertext> v;
    for (size_t i = 0; i < size; i += 2)
    {
        if (i + 1 < size) v.emplace_back(mul(cts[i], cts[i + 1]));
        else v.emplace_back(cts[i]);
    }
    mul_inplace(v);
    return v.front();
}

void Keys::mul_inplace(NativeCiphertext& a, const NativeCiphertext& b) const
{
    auto ct2_ptr = level_inplace(a, b, false);
    a *= *ct2_ptr;
    clean_level(b, ct2_ptr);
}

void Keys::mul_inplace(NativeCiphertext& a, const NativePlaintext& b) const
{
    auto pt_ptr = level_inplace(a, b, false);
    a *= *pt_ptr;
    clean_level(b, pt_ptr);
}

void Keys::mul_inplace(NativeCiphertext& a, const Scalar& b) const
{
    mul_inplace(a, encode(b, a.towers() - 1, _scale)); // default scale
}

void Keys::mul_inplace(std::vector<NativeCiphertext>& cts) const
{
    if (cts.empty()) throw "fhe::Keys::mul_inplace: No ciphertexts to multiply";
    level_inplace(cts);
    size_t size = cts.size();
    for (size_t half = (size >> 1) + (size & 1); size > 1;)
    {
        for (size_t i = half; i < size; i++) mul_inplace(cts[i - half], cts[i]);
        size = half;
        half = (half >> 1) + (half & 1);
    }
    cts.resize(1);
}

NativeCiphertext* Keys::mul_inplace(std::vector<NativeCiphertext*>& cts) const
{
    if (cts.empty()) throw "fhe::Keys::mul_inplace: No ciphertexts to multiply";
    level_inplace(cts);
    size_t size = cts.size();
    for (size_t half = (size >> 1) + (size & 1); size > 1;)
    {
        for (size_t i = half; i < size; i++) mul_inplace(*cts[i - half], *cts[i]);
        size = half;
        half = (half >> 1) + (half & 1);
    }
    cts.resize(1);
    return cts.front();
}

NativeCiphertext Keys::negate(const NativeCiphertext& ct) const
{
    return -ct;
}

void Keys::negate_inplace(NativeCiphertext& ct) const
{
    ct = -ct;
}

size_t Keys::polynomial_degree() const
{
    return n;
}

void Keys::print_rotation() const
{
    cout << "Rotation counter: ";
    for (const auto& [key, value] : rotation_counter) cout << key << ",";
    cout << endl;
}

NativeCiphertext Keys::refit(const NativeCiphertext& ct) const
{
    NativeCiphertext r = ct;
    refit_inplace(r);
    return r;
}

void Keys::refit_inplace(NativeCiphertext& ct) const
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

NativeCiphertext Keys::regularize(const NativeCiphertext& ct) const
{
    NativeCiphertext r = ct;
    regularize_inplace(r);
    return r;
}

void Keys::regularize_inplace(NativeCiphertext& ct) const
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

NativeCiphertext Keys::relinearize(const NativeCiphertext& ct) const
{
    return NativeCiphertext(ct);
}

void Keys::relinearize_inplace(NativeCiphertext&) const
{
    return; // no-op
}

void Keys::rescale_inplace(NativeCiphertext& ct) const
{
    ct.rescale();
}

void Keys::reset_rotation()
{
    rotation_counter.clear();
}

NativeCiphertext Keys::rotate(const NativeCiphertext& ct, int s)
{
    s %= slots();
    if (s < 0) s += slots();
    rotation_counter[s]++;
    return ct << s;
}

void Keys::rotate_inplace(NativeCiphertext& ct, int s)
{
    s %= slots();
    if (s < 0) s += slots();
    rotation_counter[s]++;
    ct <<= s;
}

bool Keys::save(const string& filename) const
{
    ofstream fout(filename);
    if (fout.fail()) return false;
    fout << n;
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

double Keys::scale(const NativeCiphertext& ct) const
{
    return ct.scale();
}

double Keys::scale(const NativePlaintext& pt) const
{
    return pt.scale();
}

size_t Keys::slots() const
{
    return n >> 1;
}

NativeCiphertext Keys::square(const NativeCiphertext& ct) const
{
    return ct * ct;
}

void Keys::square_inplace(NativeCiphertext& ct) const
{
    ct *= ct;
}

NativeCiphertext Keys::sub(const NativeCiphertext& a, const NativeCiphertext& b) const
{
    auto [ct1_ptr, ct2_ptr] = level(a, b, true);
    NativeCiphertext c = *ct1_ptr - *ct2_ptr;
    clean_level(a, b, ct1_ptr, ct2_ptr);
    return c;
}

NativeCiphertext Keys::sub(const NativeCiphertext& a, const NativePlaintext& b) const
{
    auto [ct_ptr, pt_ptr] = level(a, b, true);
    NativeCiphertext c = *ct_ptr - *pt_ptr;
    clean_level(a, b, ct_ptr, pt_ptr);
    return c;
}

NativeCiphertext Keys::sub(const NativeCiphertext& a, const Scalar& b) const
{
    return sub(a, encode(b, a));
}

void Keys::sub_inplace(NativeCiphertext& a, const NativeCiphertext& b) const
{
    auto ct2_ptr = level_inplace(a, b, true);
    a -= *ct2_ptr;
    clean_level(b, ct2_ptr);
}

void Keys::sub_inplace(NativeCiphertext& a, const NativePlaintext& b) const
{
    auto pt_ptr = level_inplace(a, b, true);
    a -= *pt_ptr;
    clean_level(b, pt_ptr);
}

void Keys::sub_inplace(NativeCiphertext& a, const Scalar& b) const
{
    sub_inplace(a, encode(b, a));
}

} // fhe