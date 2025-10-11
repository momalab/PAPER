#include "ciphertext.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>
#include "io.h"
#include "keys.h"
#include "plaintext.h"

using namespace std;

namespace fhe
{

inline void reduce(int& s, int slots)
{
    s %= slots;
    if (s < 0) s += slots;
}

shared_ptr<Keys> Ciphertext::def_keys = nullptr;

Ciphertext::Ciphertext(const Plaintext& a)
{
    keys = a.keys;
    ct = keys->encrypt(a.pt);
}

Ciphertext::Ciphertext(const vector<Scalar>& vs) : Ciphertext(vs, def_keys) {}

Ciphertext::Ciphertext(const vector<Scalar>& vs, const shared_ptr<Keys>& keys)
{
    if (!keys) throw invalid_argument("Ciphertext: keys is null");
    this->keys = keys;
    ct = keys->encrypt(vs);
}

Ciphertext::Ciphertext(const Scalar& a) : Ciphertext(a, def_keys) {}

Ciphertext::Ciphertext(const Scalar& v, const shared_ptr<Keys>& keys)
{
    if (!keys) throw invalid_argument("Ciphertext: keys is null");
    this->keys = keys;
    ct = keys->encrypt(v);
}

Ciphertext::operator Scalar() const
{
    return decode().front();
}

Ciphertext::operator vector<Scalar>() const
{
    return decode();
}

Ciphertext::operator Plaintext() const
{
    return decrypt();
}

// compound assignment operators

Ciphertext& Ciphertext::operator ~()
{
    keys->negate_inplace(ct);
    return *this;
}

Ciphertext& Ciphertext::operator +=(const Ciphertext& a)
{
    keys->add_inplace(this->ct, a.ct);
    return *this;
}

Ciphertext& Ciphertext::operator *=(const Ciphertext& a)
{
    if (this == &a) keys->square_inplace(this->ct);
    else keys->mul_inplace(this->ct, a.ct);
    return *this;
}

Ciphertext& Ciphertext::operator -=(const Ciphertext& a)
{
    keys->sub_inplace(this->ct, a.ct);
    return *this;
}

Ciphertext& Ciphertext::operator +=(const Plaintext& a)
{
    keys->add_inplace(this->ct, a.pt);
    return *this;
}

Ciphertext& Ciphertext::operator *=(const Plaintext& a)
{
    keys->mul_inplace(this->ct, a.pt);
    return *this;
}

Ciphertext& Ciphertext::operator -=(const Plaintext& a)
{
    keys->sub_inplace(this->ct, a.pt);
    return *this;
}

Ciphertext& Ciphertext::operator +=(const Scalar& a)
{
    keys->add_inplace(this->ct, a);
    return *this;
}

Ciphertext& Ciphertext::operator *=(const Scalar& a)
{
    keys->mul_inplace(this->ct, a);
    return *this;
}

Ciphertext& Ciphertext::operator -=(const Scalar& a)
{
    keys->sub_inplace(this->ct, a);
    return *this;
}

Ciphertext& Ciphertext::operator ^=(int e)
{
    pow(e);
    return *this;
}

Ciphertext& Ciphertext::operator <<=(int s)
{
    reduce(s, keys->slots());
    keys->rotate_inplace(ct, s);
    return *this;
}

Ciphertext& Ciphertext::operator >>=(int s)
{
    s = int(keys->slots()) - s;
    reduce(s, keys->slots());
    keys->rotate_inplace(ct, s);
    return *this;
}

// const operators

Ciphertext Ciphertext::operator -() const
{
    Ciphertext r;
    r.keys = this->keys;
    r.ct = keys->negate(this->ct);
    return r;
}

Ciphertext Ciphertext::operator +(const Ciphertext& a) const
{
    Ciphertext r;
    r.keys = this->keys;
    r.ct = keys->add(this->ct, a.ct);
    return r;
}

Ciphertext Ciphertext::operator *(const Ciphertext& a) const
{
    Ciphertext r;
    r.keys = this->keys;
    if (this == &a) r.ct = keys->square(this->ct);
    else r.ct = keys->mul(this->ct, a.ct);
    return r;
}

Ciphertext Ciphertext::operator -(const Ciphertext& a) const
{
    Ciphertext r;
    r.keys = this->keys;
    r.ct = keys->sub(this->ct, a.ct);
    return r;
}

Ciphertext Ciphertext::operator +(const Plaintext& a) const
{
    Ciphertext r;
    r.keys = this->keys;
    r.ct = keys->add(this->ct, a.pt);
    return r;
}

Ciphertext Ciphertext::operator *(const Plaintext& a) const
{
    Ciphertext r;
    r.keys = this->keys;
    r.ct = keys->mul(this->ct, a.pt);
    return r;
}

Ciphertext Ciphertext::operator -(const Plaintext& a) const
{
    Ciphertext r;
    r.keys = this->keys;
    r.ct = keys->sub(this->ct, a.pt);
    return r;
}

Ciphertext Ciphertext::operator +(const Scalar& a) const
{
    Ciphertext r;
    r.keys = this->keys;
    r.ct = keys->add(this->ct, a);
    return r;
}

Ciphertext Ciphertext::operator *(const Scalar& a) const
{
    Ciphertext r;
    r.keys = this->keys;
    r.ct = keys->mul(this->ct, a);
    return r;
}

Ciphertext Ciphertext::operator -(const Scalar& a) const
{
    Ciphertext r;
    r.keys = this->keys;
    r.ct = keys->sub(this->ct, a);
    return r;
}

Ciphertext Ciphertext::operator ^(int e) const
{
    Ciphertext r(*this);
    r.pow(e);
    return r;
}

Ciphertext Ciphertext::operator <<(int s) const
{
    reduce(s, keys->slots());
    Ciphertext r;
    r.keys = this->keys;
    r.ct = keys->rotate(this->ct, s);
    return r;
}

Ciphertext Ciphertext::operator >>(int s) const
{
    s = int(keys->slots()) - s;
    reduce(s, keys->slots());
    Ciphertext r;
    r.keys = this->keys;
    r.ct = keys->rotate(this->ct, s);
    return r;
}

// friend operators

Ciphertext operator +(const Plaintext& a, const Ciphertext& b)
{
    return b + a;
}

Ciphertext operator *(const Plaintext& a, const Ciphertext& b)
{
    return b * a;
}

Ciphertext operator -(const Plaintext& a, const Ciphertext& b)
{
    return -b + a;
}

Ciphertext operator +(const Scalar& a, const Ciphertext& b)
{
    return b + a;
}

Ciphertext operator *(const Scalar& a, const Ciphertext& b)
{
    return b * a;
}

Ciphertext operator -(const Scalar& a, const Ciphertext& b)
{
    return -b + a;
}

// functions

Ciphertext Ciphertext::add(const vector<Ciphertext> & v, const shared_ptr<Keys>& keys)
{
    if (v.empty()) return Ciphertext(0, keys);
    Ciphertext r;
    r.keys = v.front().keys;
    vector<NativeCiphertext> vct;
    for (auto & e : v) vct.push_back(e.ct);
    r.ct = r.keys->add(vct);
    return r;
}

Ciphertext& Ciphertext::add_inplace(vector<Ciphertext> & v, const shared_ptr<Keys>& keys)
{
    if (v.empty()) v.emplace_back(Ciphertext(0, keys));
    if (v.size() == 1) return v.front();
    vector<NativeCiphertext*> vct;
    for (auto & e : v) vct.push_back(&e.ct);
    auto& k = v.front().keys;
    k->add_inplace(vct);
    v.resize(1);
    return v.front();
}

Ciphertext& Ciphertext::addslots_inplace(Ciphertext& ct, size_t s)
{
    while(s >>= 1) ct += ct << s;
    return ct;
}

vector<Scalar> Ciphertext::decode() const
{
    return keys->decode(ct);
}

Plaintext Ciphertext::decrypt() const
{
    Plaintext r;
    r.keys = keys;
    r.pt = keys->decrypt(ct);
    return r;
}

const Keys& Ciphertext::default_keys(const Keys& keys)
{
    def_keys = make_shared<Keys>(keys);
    return *def_keys;
}

const Keys& Ciphertext::default_keys(const std::shared_ptr<Keys>& keys)
{
    if (!keys) throw invalid_argument("Ciphertext: keys is null");
    Ciphertext::def_keys = keys;
    return *keys;
}

size_t Ciphertext::default_polynomial_degree()
{
    return def_keys->polynomial_degree();
}

size_t Ciphertext::default_slots()
{
    return def_keys->slots();
}

tuple<vector<size_t>,vector<size_t>> Ciphertext::indices_and_lengths(const vector<bool> & pattern)
{
    return Plaintext::indices_and_lengths(pattern);
}

double Ciphertext::keyscale() const
{
    return keys->scale();
}

int Ciphertext::level() const
{
    return keys->level(ct);
}

void Ciphertext::modswitch_inplace(int lvl)
{
    if (lvl < 0) lvl = level() - 1;
    while (level() > lvl) keys->modswitch_inplace(ct);
}

size_t Ciphertext::polynomial_degree() const
{
    return keys->polynomial_degree();
}

void Ciphertext::pow(int e)
{
    if (e < 0) throw "Negative exponent not supported";
    if (e == 0) ct = keys->encrypt(Scalar(1));
    if (e == 1) return;

    NativeCiphertext copy = ct;
    for (int i = 1; i < e; i++)
    {
        keys->regularize_inplace(this->ct);
        keys->mul_inplace(ct, copy);
    }
}

uint64_t Ciphertext::qi() const
{
    return keys->modulus(level());
}

void Ciphertext::refit_inplace()
{
    keys->refit_inplace(ct);
}

void Ciphertext::regularize_inplace()
{
    keys->regularize_inplace(ct);
}

void Ciphertext::relinearize_inplace()
{
    keys->relinearize_inplace(ct);
}

double Ciphertext::scale() const
{
    return keys->scale(ct);
}

void Ciphertext::shiftleft_inplace(vector<Ciphertext>& v, int s, Scalar scale)
{
    if (v.empty()) return;
    size_t slots = v.front().keys->slots();
    size_t size = v.size();
    int shift_mod = size * slots;
    size_t shift_amount = (s % shift_mod + shift_mod) % shift_mod;
    size_t sdiv = shift_amount / slots;
    size_t smod = shift_amount % slots;
    if (size == 1)
    {
        if (smod) v.front() <<= smod;
        if (abs(scale - 1.0) > 1e-8) v.front() *= scale;
        return;
    }
    std::rotate(v.begin(), v.begin() + sdiv, v.end());
    if (smod)
    {
        Plaintext mask0, mask1;
        {
            vector<Scalar> v0(slots, 0), v1(slots, scale);
            for (size_t i = slots - 1; i > slots - 1 - smod; i--)
            {
                v0[i] = scale;
                v1[i] = 0.0;
            }
            mask0 = move(Plaintext(v0, v[0].level(), v[0].keyscale()));
            mask1 = move(Plaintext(v1, v[0].level(), v[0].keyscale()));
        }
        for (auto& ct : v) ct <<= smod;

        auto v0 = v[0];
        for (size_t i = 0; i < size - 1; i++) v[i] = v[i] * mask1 + v[i + 1] * mask0;
        v[size - 1] = v[size - 1] * mask1 + v0 * mask0;
    }
    else
    {
        Plaintext pt(scale, v[0].level(), v[0].keyscale());
        for (auto& e : v) e *= pt;
    }
}

void Ciphertext::shiftleft_reformat_inplace(vector<Ciphertext> & v, int s, const vector<bool> & pattern, const Scalar& scale)
{
    if (v.empty()) return;
    size_t slots = v.front().keys->slots();
    size_t size = v.size();
    int shift_mod = size * slots;
    size_t shift_amount = (s % shift_mod + shift_mod) % shift_mod;
    size_t smod = shift_amount % slots;
    auto [indices, lengths] = Plaintext::indices_and_lengths(pattern);
    map<size_t,Plaintext> masks = Plaintext::create_masks(indices, lengths, scale);

    size_t o = 0;
    vector<Ciphertext> vct;
    for (size_t i = 0, shift = 0; i < indices.size(); i++)
    {
        auto& idx = indices[i];
        auto& len = lengths[i];
        size_t ict = idx / slots;
        size_t slot = idx % slots;
        size_t key = slot * slots + len;
        auto& mask = masks[key];
        auto& ct = v[ict];
        vct.push_back(ct * mask);
        shift = (i ? shift + idx - indices[i-1] - lengths[i-1] : smod) % slots;
        vct.back() <<= shift;
        if ((slot + len - shift) % slots == 0 || i == indices.size() - 1)
        {
            v[o++] = Ciphertext::add(vct);
            vct.clear();
        }
    }
    v.resize(o);
}

size_t Ciphertext::slots() const
{
    return keys->slots();
}

ostream & operator <<(ostream & os, const Ciphertext& ct)
{
    os << ct.decode();
    return os;
}

} // fhe