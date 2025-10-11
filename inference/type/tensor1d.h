#pragma once

#include <initializer_list>
#include <ostream>
#include <vector>
#include "tensor1t.h"

namespace type
{

template <>
class Tensor<1,double>
{
    private:
        std::vector<double> _data;
        static double _precision;

    public:
        Tensor() = default;
        Tensor(const Tensor<1,double>&) = default;
        Tensor(Tensor<1,double>&&) noexcept = default;
        // Tensor(const std::vector<double>&);
        Tensor(const std::initializer_list<std::size_t>&, const double& = 0.0);
        Tensor(const std::vector<std::size_t>&, const double& = 0.0);
        Tensor(std::vector<std::size_t>::const_iterator, std::vector<std::size_t>::const_iterator, const double& = 0.0);
        ~Tensor() = default;

        Tensor<1,double>& operator=(const Tensor<1,double>&) = default;
        Tensor<1,double>& operator=(Tensor<1,double>&&) noexcept = default;

        double& operator[](int i);
        const double& operator[](int i) const;

        Tensor<1,double>& operator+=(const Tensor<1,double>&);
        Tensor<1,double>& operator-=(const Tensor<1,double>&);
        Tensor<1,double>& operator*=(const Tensor<1,double>&);
        Tensor<1,double> operator+(const Tensor<1,double>&) const;
        Tensor<1,double> operator-(const Tensor<1,double>&) const;
        Tensor<1,double> operator*(const Tensor<1,double>&) const;
        Tensor<1,double>& operator+=(double);
        Tensor<1,double>& operator-=(double);
        Tensor<1,double>& operator*=(double);
        Tensor<1,double> operator+(double) const;
        Tensor<1,double> operator-(double) const;
        Tensor<1,double> operator*(double) const;
        bool operator==(const Tensor<1,double>&) const;
        bool operator!=(const Tensor<1,double>&) const;
        bool operator==(double) const;
        bool operator!=(double) const;

        bool any_empty() const;
        std::size_t argmax() const;
        double& back();
        const double& back() const;
        typename std::vector<double>::iterator begin();
        typename std::vector<double>::const_iterator begin() const;
        void emplace_back(const double&);
        void emplace_back(double&&);
        bool empty() const;
        typename std::vector<double>::iterator end();
        typename std::vector<double>::const_iterator end() const;
        double& front();
        const double& front() const;
        bool front_empty() const;
        void push_back(const double&);
        void push_back(double&&);
        void resize(std::size_t);
        void resize(std::size_t, const double&);
        std::size_t size() const;
        std::vector<std::size_t> shape() const;
        std::vector<double>& vector();
        const std::vector<double>& vector() const;

        static Tensor<1,double> copy(const std::vector<double>&);
        template <class U> static Tensor<1,double> copy(const std::vector<U>&);
        static Tensor<1,double> move(std::vector<double>&&);
        template <class U> static Tensor<1,double> move(std::vector<U>&&);
        static double precision();
        static double precision(double);
};

} // type

#include "tensor1d.hpp"