#include "native.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#include "vectorization.h"

using namespace std;
using namespace util;

namespace fhe
{
// NativePlaintext

NativePlaintext::NativePlaintext(const vector<Scalar>& vs, size_t slots, const vector<int>& logq, double scale)
{
    _scale = scale;

    const auto& towers = logq.size();
    _qi.resize(towers);
    for (size_t i = 0; i < towers; i++) _qi[i] = pow(2.0, logq[i]);

    _data = vs;
    _data.resize(slots, 0.0);
    for (auto& e : _data) e *= scale;
}

NativePlaintext::NativePlaintext(const vector<Scalar>& vs, size_t slots, const vector<double>& moduli, double scale)
{
    _scale = scale;
    _qi = moduli;
    _data = vs;
    _data.resize(slots, 0.0);
    for (auto& e : _data) e *= scale;
}

NativePlaintext::NativePlaintext(size_t slots, const vector<int>& logq, double scale)
    : NativePlaintext(vector<Scalar>(), slots, logq, scale){}

NativePlaintext::NativePlaintext(size_t slots, const vector<double>& moduli, double scale)
    : NativePlaintext(vector<Scalar>(), slots, moduli, scale){}

NativePlaintext::NativePlaintext(const Scalar& value, size_t slots, const vector<int>& logq, double scale)
    : NativePlaintext(vector<Scalar>(slots, value), slots, logq, scale){}

NativePlaintext::NativePlaintext(const Scalar& value, size_t slots, const vector<double>& moduli, double scale)
    : NativePlaintext(vector<Scalar>(slots, value), slots, moduli, scale){}

NativePlaintext::operator vector<Scalar>() const
{
    auto vs = _data;
    for (auto& e : vs) e /= _scale;
    return vs;
}

NativePlaintext& NativePlaintext::operator+=(const NativePlaintext& a)
{
    if (!match(a)) throw runtime_error("NativePlaintext::operator+=: size mismatch");
    if (_scale != a._scale) throw runtime_error("NativePlaintext::operator+=: scale mismatch (" + to_string(_scale) + " != " + to_string(a._scale) + ")");
    _data += a._data;
    return *this;
}

NativePlaintext& NativePlaintext::operator-=(const NativePlaintext& a)
{
    if (!match(a)) throw runtime_error("NativePlaintext::operator-=: size mismatch");
    if (_scale != a._scale) throw runtime_error("NativePlaintext::operator-=: scale mismatch");
    _data -= a._data;
    return *this;
}

NativePlaintext& NativePlaintext::operator*=(const NativePlaintext& a)
{
    if (!match(a)) throw runtime_error("NativePlaintext::operator*=: size mismatch");
    _data *= a._data;
    _scale *= a._scale;
    return *this;
}

NativePlaintext& NativePlaintext::operator<<=(int s)
{
    s %= slots();
    if (s < 0) s += slots();
    rotate(_data.begin(), _data.begin() + s, _data.end());
    return *this;
}

NativePlaintext NativePlaintext::operator+(const NativePlaintext& a) const
{
    if (!match(a)) throw runtime_error("NativePlaintext::operator+: size mismatch");
    if (_scale != a._scale) throw runtime_error("NativePlaintext::operator+: scale mismatch");
    NativePlaintext r = *this;
    r += a;
    return r;
}

NativePlaintext NativePlaintext::operator-() const
{
    NativePlaintext r = *this;
    for (auto& e : r._data) e = -e;
    return r;
}

NativePlaintext NativePlaintext::operator-(const NativePlaintext& a) const
{
    if (!match(a)) throw runtime_error("NativePlaintext::operator-: size mismatch");
    if (_scale != a._scale) throw runtime_error("NativePlaintext::operator-: scale mismatch");
    NativePlaintext r = *this;
    r -= a;
    return r;
}

NativePlaintext NativePlaintext::operator*(const NativePlaintext& a) const
{
    if (!match(a)) throw runtime_error("NativePlaintext::operator*: size mismatch");
    NativePlaintext r = *this;
    r *= a;
    return r;
}

NativePlaintext NativePlaintext::operator<<(int s) const
{
    NativePlaintext r = *this;
    r <<= s;
    return r;
}

bool NativePlaintext::match(const NativePlaintext& a) const
{
    return (slots() == a.slots() && towers() == a.towers());
}

void NativePlaintext::modswitch()
{
    if (_qi.size() <= 1) throw runtime_error("NativePlaintext::modswitch: no towers left");
    _qi.pop_back();
}

const vector<double>& NativePlaintext::moduli() const
{
    return _qi;
}

void NativePlaintext::rescale()
{
    if (_qi.size() <= 1) throw runtime_error("NativePlaintext::rescale: no towers left (" + to_string(_qi.size()) + ")");
    for (auto& e : _data) e /= _qi.back();
    _scale /= _qi.back();
    _qi.pop_back();
    if (round(_scale) <= 1.0) throw runtime_error("NativePlaintext::rescale: scale too small");
}

double NativePlaintext::scale() const
{
    return _scale;
}

size_t NativePlaintext::slots() const
{
    return _data.size();
}

size_t NativePlaintext::towers() const
{
    return _qi.size();
}

// NativeCiphertext

NativeCiphertext::NativeCiphertext(const NativePlaintext& pt)
{
    data = pt;
}

NativeCiphertext::NativeCiphertext(const vector<Scalar>& vs, size_t slots, const vector<int>& logq, double scale)
{
    data = NativePlaintext(vs, slots, logq, scale);
}

NativeCiphertext::NativeCiphertext(const vector<Scalar>& vs, size_t slots, const vector<double>& moduli, double scale)
{
    data = NativePlaintext(vs, slots, moduli, scale);
}

NativeCiphertext::NativeCiphertext(size_t slots, const vector<int>& logq, double scale)
{
    data = NativePlaintext(slots, logq, scale);
}

NativeCiphertext::NativeCiphertext(size_t slots, const vector<double>& moduli, double scale)
{
    data = NativePlaintext(slots, moduli, scale);
}

NativeCiphertext::NativeCiphertext(const Scalar& value, size_t slots, const vector<int>& logq, double scale)
{
    data = NativePlaintext(value, slots, logq, scale);
}

NativeCiphertext::NativeCiphertext(const Scalar& value, size_t slots, const vector<double>& moduli, double scale)
{
    data = NativePlaintext(value, slots, moduli, scale);
}

NativeCiphertext::operator NativePlaintext() const
{
    return NativePlaintext(data);
}

NativeCiphertext::operator vector<Scalar>() const
{
    return vector<Scalar>(data);
}

NativeCiphertext& NativeCiphertext::operator+=(const NativeCiphertext& a)
{
    data += a.data;
    return *this;
}

NativeCiphertext& NativeCiphertext::operator+=(const NativePlaintext& a)
{
    data += a;
    return *this;
}

NativeCiphertext& NativeCiphertext::operator-=(const NativeCiphertext& a)
{
    data -= a.data;
    return *this;
}

NativeCiphertext& NativeCiphertext::operator-=(const NativePlaintext& a)
{
    data -= a;
    return *this;
}

NativeCiphertext& NativeCiphertext::operator*=(const NativeCiphertext& a)
{
    data *= a.data;
    return *this;
}

NativeCiphertext& NativeCiphertext::operator*=(const NativePlaintext& a)
{
    data *= a;
    return *this;
}

NativeCiphertext& NativeCiphertext::operator<<=(int s)
{
    data <<= s;
    return *this;
}

NativeCiphertext NativeCiphertext::operator+(const NativeCiphertext& a) const
{
    NativeCiphertext r;
    r.data = data + a.data;
    return r;
}

NativeCiphertext NativeCiphertext::operator-() const
{
    NativeCiphertext r;
    r.data = -data;
    return r;
}

NativeCiphertext NativeCiphertext::operator+(const NativePlaintext& a) const
{
    NativeCiphertext r;
    r.data = data + a;
    return r;
}

NativeCiphertext NativeCiphertext::operator-(const NativeCiphertext& a) const
{
    NativeCiphertext r;
    r.data = data - a.data;
    return r;
}

NativeCiphertext NativeCiphertext::operator-(const NativePlaintext& a) const
{
    NativeCiphertext r;
    r.data = data - a;
    return r;
}

NativeCiphertext NativeCiphertext::operator*(const NativeCiphertext& a) const
{
    NativeCiphertext r;
    r.data = data * a.data;
    return r;
}

NativeCiphertext NativeCiphertext::operator*(const NativePlaintext& a) const
{
    NativeCiphertext r;
    r.data = data * a;
    return r;
}

NativeCiphertext NativeCiphertext::operator<<(int s) const
{
    NativeCiphertext r = *this;
    r.data <<= s;
    return r;
}


bool NativeCiphertext::match(const NativeCiphertext& a) const
{
    return data.match(a.data);
}

void NativeCiphertext::modswitch()
{
    data.modswitch();
}

const vector<double>& NativeCiphertext::moduli() const
{
    return data.moduli();
}

void NativeCiphertext::rescale()
{
    data.rescale();
}

double NativeCiphertext::scale() const
{
    return data.scale();
}

size_t NativeCiphertext::slots() const
{
    return data.slots();
}

size_t NativeCiphertext::towers() const
{
    return data.towers();
}

} // fhe