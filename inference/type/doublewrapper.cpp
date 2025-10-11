#include "doublewrapper.h"

#include <atomic>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>

using namespace std;

atomic<int> DoubleWrapper::id_counter = -1;
atomic<int> DoubleWrapper::pt_counter = -1;
unordered_map<string,int> DoubleWrapper::plaintexts;
size_t DoubleWrapper::nslots = 1<<14;
ofstream DoubleWrapper::log("operations.log");
ofstream DoubleWrapper::clog("plaintexts.log");
mutex DoubleWrapper::mtx_log;
mutex DoubleWrapper::mtx_clog;

inline string stringfy(const double a)
{
    return to_string(a);
}

inline string stringfy(const vector<double> & a)
{
    string s = "{";
    if (!a.empty()) s += stringfy(a.front());
    for (size_t i = 1; i < a.size(); i++) s += "," + stringfy(a[i]);
    s += "}";
    return s;
}

DoubleWrapper::DoubleWrapper()
{
    id = ++id_counter;
    v.resize(nslots, 0);
}

DoubleWrapper::DoubleWrapper(double a)
{
    id = ++id_counter;
    v.resize(nslots, a);
    stringstream ss;
    ss << ">C" << id << " = " << a << '\n';
    write_log(ss);
}

DoubleWrapper::DoubleWrapper(const vector<double> & a)
{
    id = ++id_counter;
    v.assign(a.begin(), a.end());
    stringstream ss;
    ss << ">C" << id << " = " << "{";
    if (!a.empty()) ss << a.front();
    for (size_t i = 1; i < a.size(); i++) ss << "," << a[i];
    ss << "}\n";
}

DoubleWrapper::operator vector<double>() const
{
    stringstream ss;
    ss << "<C" << id << '\n';
    write_log(ss);
    return v;
}

DoubleWrapper & DoubleWrapper::operator+=(const DoubleWrapper & rhs)
{
    for (size_t i = 0; i < nslots; i++) v[i] += rhs.v[i];
    stringstream ss;
    int new_id = ++id_counter;
    ss << "C" << new_id << " = C" << id << " + C" << rhs.id << '\n';
    id = new_id;
    write_log(ss);
    return *this;
}

DoubleWrapper & DoubleWrapper::operator-=(const DoubleWrapper & rhs)
{
    for (size_t i = 0; i < nslots; i++) v[i] -= rhs.v[i];
    stringstream ss;
    int new_id = ++id_counter;
    ss << "C" << new_id << " = C" << id << " - C" << rhs.id << '\n';
    write_log(ss);
    return *this;
}

DoubleWrapper & DoubleWrapper::operator*=(const DoubleWrapper & rhs)
{
    for (size_t i = 0; i < nslots; i++) v[i] *= rhs.v[i];
    stringstream ss;
    int new_id = ++id_counter;
    ss << "C" << new_id << " = C" << id << " * C" << rhs.id << '\n';
    id = new_id;
    write_log(ss);
    return *this;
}

DoubleWrapper & DoubleWrapper::operator+=(const double rhs)
{
    for (size_t i = 0; i < nslots; i++) v[i] += rhs;
    stringstream ss;
    int new_id = ++id_counter;
    ss << "C" << new_id << " = C" << id << " + P" << pt(rhs) << '\n';
    id = new_id;
    write_log(ss);
    return *this;
}

DoubleWrapper & DoubleWrapper::operator-=(const double rhs)
{
    for (size_t i = 0; i < nslots; i++) v[i] += rhs;
    stringstream ss;
    int new_id = ++id_counter;
    ss << "C" << new_id << " = C" << id << " - P" << pt(rhs) << '\n';
    id = new_id;
    write_log(ss);
    return *this;
}

DoubleWrapper & DoubleWrapper::operator*=(const double rhs)
{
    for (size_t i = 0; i < nslots; i++) v[i] *= rhs;
    stringstream ss;
    int new_id = ++id_counter;
    ss << "C" << new_id << " = C" << id << " * P" << pt(rhs) << '\n';
    id = new_id;
    write_log(ss);
    return *this;
}

DoubleWrapper DoubleWrapper::operator-() const
{
    DoubleWrapper dw;
    for (size_t i = 0; i < nslots; i++) dw.v[i] = -v[i];
    stringstream ss;
    ss << "C" << dw.id << " = P" << pt(0) << " - C" << id << '\n';
    write_log(ss);
    return dw;
}

DoubleWrapper DoubleWrapper::operator+(const DoubleWrapper & rhs) const
{
    DoubleWrapper dw;
    for (size_t i = 0; i < nslots; i++) dw.v[i] = v[i] + rhs.v[i];
    stringstream ss;
    ss << "C" << dw.id << " = C" << id << " + C" << rhs.id << '\n';
    write_log(ss);
    return dw;
}

DoubleWrapper DoubleWrapper::operator-(const DoubleWrapper & rhs) const
{
    DoubleWrapper dw;
    for (size_t i = 0; i < nslots; i++) dw.v[i] = v[i] - rhs.v[i];
    stringstream ss;
    ss << "C" << dw.id << " = C" << id << " - " << "C" << rhs.id << '\n';
    write_log(ss);
    return dw;
}

DoubleWrapper DoubleWrapper::operator*(const DoubleWrapper & rhs) const
{
    DoubleWrapper dw;
    for (size_t i = 0; i < nslots; i++) dw.v[i] = v[i] * rhs.v[i];
    stringstream ss;
    ss << "C" << dw.id << " = C" << id << " * " << "C" << rhs.id << '\n';
    write_log(ss);
    return dw;
}

DoubleWrapper DoubleWrapper::operator+(const double rhs) const
{
    DoubleWrapper dw;
    for (size_t i = 0; i < nslots; i++) dw.v[i] = v[i] + rhs;
    stringstream ss;
    ss << "C" << dw.id << " = C" << id << " + P" << pt(rhs) << '\n';
    write_log(ss);
    return dw;
}

DoubleWrapper DoubleWrapper::operator-(const double rhs) const
{
    DoubleWrapper dw;
    for (size_t i = 0; i < nslots; i++) dw.v[i] = v[i] - rhs;
    stringstream ss;
    ss << "C" << dw.id << " = C" << id << " - P" << pt(rhs) << '\n';
    write_log(ss);
    return dw;
}

DoubleWrapper DoubleWrapper::operator*(const double rhs) const
{
    DoubleWrapper dw;
    for (size_t i = 0; i < nslots; i++) dw.v[i] = v[i] * rhs;
    stringstream ss;
    ss << "C" << dw.id << " = C" << id << " * P" << pt(rhs) << '\n';
    write_log(ss);
    return dw;
}

DoubleWrapper operator+(const double lhs, const DoubleWrapper & rhs)
{
    return rhs + lhs;
}

DoubleWrapper operator-(const double lhs, const DoubleWrapper & rhs)
{
    DoubleWrapper dw;
    for (size_t i = 0; i < DoubleWrapper::slots(); i++) dw.v[i] = lhs - rhs.v[i];
    stringstream ss;
    ss << "C" << dw.id << " = P" << DoubleWrapper::pt(lhs) << " - " << "C" << rhs.id << '\n';
    DoubleWrapper::write_log(ss);
    return dw;
}

DoubleWrapper operator*(const double lhs, const DoubleWrapper & rhs)
{
    return rhs * lhs;
}

vector<double> DoubleWrapper::get() const
{
    return vector<double>(*this);
}

int DoubleWrapper::pt(const double a)
{
    auto key = stringfy(a);
    stringstream ss;
    if (plaintexts.find(key) == plaintexts.end())
    {
        plaintexts[key] = ++pt_counter;
        ss << "P" << plaintexts[key] << " = " << key << '\n';
    }
    write_clog(ss);
    return plaintexts[key];
}

int DoubleWrapper::pt(const vector<double> & a)
{
    auto key = stringfy(a);
    stringstream ss;
    if (plaintexts.find(key) == plaintexts.end())
    {
        plaintexts[key] = ++pt_counter;
        ss << "P" << plaintexts[key] << " = " << key << '\n';
    }
    write_clog(ss);
    return plaintexts[key];
}

size_t DoubleWrapper::slots()
{
    return nslots;
}

void DoubleWrapper::slots(size_t slots)
{
    nslots = slots;
}

std::ostream & operator<<(std::ostream & os, const DoubleWrapper & dw)
{
    os << "[ " << dw.v.front() << " ... " << dw.v.back() << " ]";
    return os;
}