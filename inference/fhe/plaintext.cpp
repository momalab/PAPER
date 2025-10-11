#include "plaintext.h"

#include <map>
#include <memory>
#include <tuple>
#include <vector>
#include "ciphertext.h"
#include "io.h"
#include "keys.h"

using namespace std;

namespace fhe
{

shared_ptr<Keys> Plaintext::def_keys = nullptr;


Plaintext::Plaintext(const Scalar& scalar, const Ciphertext*& ct)
{
    if (ct)
    {
        keys = ct->keys;
        pt = keys->encode(scalar, ct->ct);
    }
    else if (def_keys)
    {
        keys = def_keys;
        pt = keys->encode(scalar);
    }
    else throw invalid_argument("Plaintext: Ciphertext and default keys are null");
}

Plaintext::Plaintext(const Scalar& scalar, int level, double scale)
{
    if (!def_keys) throw invalid_argument("Plaintext: default keys are null");
    if (!scale) scale = def_keys->scale();
    this->keys = def_keys;
    pt = keys->encode(scalar, level, scale);
}

Plaintext::Plaintext(const Scalar& scalar, const shared_ptr<Keys>& keys)
{
    if (!keys) throw invalid_argument("Plaintext: keys is null");
    this->keys = keys;
    pt = keys->encode(scalar);
}

Plaintext::Plaintext(const vector<Scalar>& vs, const Ciphertext*& ct)
{
    if (ct)
    {
        keys = ct->keys;
        pt = keys->encode(vs, ct->ct);
    }
    else if (def_keys)
    {
        keys = def_keys;
        pt = keys->encode(vs);
    }
    else throw invalid_argument("Plaintext: Ciphertext and default keys are null");
}

Plaintext::Plaintext(const vector<Scalar>& vs, int level, double scale)
{
    if (!def_keys) throw invalid_argument("Plaintext: default keys are null");
    if (!scale) scale = def_keys->scale();
    this->keys = def_keys;
    pt = keys->encode(vs, level, scale);
}

Plaintext::Plaintext(const vector<Scalar>& vs, const shared_ptr<Keys>& keys)
{
    if (!keys) throw invalid_argument("Plaintext: keys is null");
    this->keys = keys;
    pt = keys->encode(vs);
}

Plaintext::operator Scalar()
{
    return decode().front();
}

Plaintext::operator vector<Scalar>() const
{
    return decode();
}

map<size_t,Plaintext> Plaintext::create_masks(const vector<size_t> & indices, const vector<size_t> & lengths, const Scalar& scaling)
{
    size_t slots = Plaintext::default_slots();
    map<size_t,Plaintext> masks;

    for (size_t i = 0; i < indices.size(); i++)
    {
        size_t slot = indices[i] % slots;
        size_t key = slot * slots + lengths[i];
        if (masks.find(key) == masks.end()) // generate mask if not found
        {
            vector<Scalar> v(slots, 0);
            for (size_t j = 0; j < lengths[i]; j++) v[slot+j] = scaling;
            masks.emplace(key, Plaintext(v));
        }
    }

    return masks;
}

vector<Scalar> Plaintext::decode() const
{
    return keys->decode(pt);
}

const Keys& Plaintext::default_keys(const Keys& keys)
{
    def_keys = make_shared<Keys>(keys);
    return *def_keys;
}

const Keys& Plaintext::default_keys(const std::shared_ptr<Keys>& keys)
{
    if (!keys) throw invalid_argument("Plaintext: keys is null");
    Plaintext::def_keys = keys;
    return *keys;
}

size_t Plaintext::default_polynomial_degree()
{
    return def_keys->polynomial_degree();
}

size_t Plaintext::default_slots()
{
    return def_keys->slots();
}

tuple<vector<size_t>,vector<size_t>> Plaintext::indices_and_lengths(const vector<bool> & pattern)
{
    size_t slots = Plaintext::default_slots();
    size_t nCT = pattern.size() / slots + (pattern.size() % slots != 0);
    vector<size_t> indices, lengths;
    for (size_t ict = 0, oslot = 0; ict < nCT; ict++)
    {
        for (size_t slot = 0, first = 1; slot < slots; slot++)
        {
            size_t idx = ict * slots + slot;
            if (pattern[idx])
            {
                if (oslot++ == slots || first || idx != indices.back() + lengths.back())
                {
                    indices.push_back(idx);
                    lengths.push_back(1);
                    if (oslot > slots) oslot = 1;
                    first = 0;
                }
                else lengths.back()++;
            }
        }
    }
    return make_tuple(move(indices), move(lengths));
}

double Plaintext::keyscale() const
{
    return keys->scale();
}

int Plaintext::level() const
{
    return keys->level(pt);
}

size_t Plaintext::polynomial_degree() const
{
    return keys->polynomial_degree();
}

double Plaintext::scale() const
{
    return keys->scale(pt);
}

size_t Plaintext::slots() const
{
    return keys->slots();
}

ostream & operator <<(ostream & os, const Plaintext & pt)
{
    os << pt.decode();
    return os;
}

} // fhe