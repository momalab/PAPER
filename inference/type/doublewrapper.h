#pragma once

#include <atomic>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>

class DoubleWrapper
{
    private:
        static std::atomic<int> id_counter;
        static std::atomic<int> pt_counter;
        static std::unordered_map<std::string,int> plaintexts;
        static size_t nslots;
        static std::ofstream log;
        static std::ofstream clog;
        static std::mutex mtx_log;
        static std::mutex mtx_clog;
        static int pt(const double);
        static int pt(const std::vector<double> &);
        static void write_clog(const std::stringstream &);
        static void write_log(const std::stringstream &);

        int id;
        std::vector<double> v;

    public:
        DoubleWrapper();
        DoubleWrapper(double a);
        DoubleWrapper(const std::vector<double> & a);
        DoubleWrapper(const DoubleWrapper &) = default;
        DoubleWrapper(DoubleWrapper &&) = default;

        operator std::vector<double>() const;

        DoubleWrapper & operator=(const DoubleWrapper &) = default;
        DoubleWrapper & operator+=(const DoubleWrapper &);
        DoubleWrapper & operator-=(const DoubleWrapper &);
        DoubleWrapper & operator*=(const DoubleWrapper &);
        DoubleWrapper & operator+=(const double);
        DoubleWrapper & operator-=(const double);
        DoubleWrapper & operator*=(const double);

        DoubleWrapper operator-() const;
        DoubleWrapper operator+(const DoubleWrapper &) const;
        DoubleWrapper operator-(const DoubleWrapper &) const;
        DoubleWrapper operator*(const DoubleWrapper &) const;
        DoubleWrapper operator+(const double) const;
        DoubleWrapper operator-(const double) const;
        DoubleWrapper operator*(const double) const;

        std::vector<double> get() const;
        static size_t slots();
        static void slots(size_t slots);

        friend DoubleWrapper operator+(const double, const DoubleWrapper &);
        friend DoubleWrapper operator-(const double, const DoubleWrapper &);
        friend DoubleWrapper operator*(const double, const DoubleWrapper &);
        friend std::ostream & operator<<(std::ostream &, const DoubleWrapper &);
};

inline void DoubleWrapper::write_clog(const std::stringstream & ss)
{
    mtx_clog.lock();
    clog << ss.rdbuf();
    mtx_clog.unlock();
}

inline void DoubleWrapper::write_log(const std::stringstream & ss)
{
    mtx_clog.lock();
    log << ss.rdbuf();
    mtx_clog.unlock();
}