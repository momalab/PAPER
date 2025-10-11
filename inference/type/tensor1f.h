#pragma once

#include <initializer_list>
#include <ostream>
#include <vector>
#include "tensor1t.h"

namespace type
{

template <>
class Tensor<1,float>
{
    private:
        std::vector<float> _data;
        static float _precision;

    public:
        Tensor() = default;
        Tensor(const Tensor<1,float>&) = default;
        Tensor(Tensor<1,float>&&) noexcept = default;
        // Tensor(const std::vector<float>&);
        Tensor(const std::initializer_list<std::size_t>&, const float& = 0.0f);
        Tensor(const std::vector<std::size_t>&, const float& = 0.0f);
        Tensor(std::vector<std::size_t>::const_iterator, std::vector<std::size_t>::const_iterator, const float& = 0.0f);
        ~Tensor() = default;

        Tensor<1,float>& operator=(const Tensor<1,float>&) = default;
        Tensor<1,float>& operator=(Tensor<1,float>&&) noexcept = default;

        float& operator[](int i);
        const float& operator[](int i) const;

        Tensor<1,float>& operator+=(const Tensor<1,float>&);
        Tensor<1,float>& operator-=(const Tensor<1,float>&);
        Tensor<1,float>& operator*=(const Tensor<1,float>&);
        Tensor<1,float> operator+(const Tensor<1,float>&) const;
        Tensor<1,float> operator-(const Tensor<1,float>&) const;
        Tensor<1,float> operator*(const Tensor<1,float>&) const;
        Tensor<1,float>& operator+=(float);
        Tensor<1,float>& operator-=(float);
        Tensor<1,float>& operator*=(float);
        Tensor<1,float> operator+(float) const;
        Tensor<1,float> operator-(float) const;
        Tensor<1,float> operator*(float) const;
        bool operator==(const Tensor<1,float>&) const;
        bool operator!=(const Tensor<1,float>&) const;
        bool operator==(float) const;
        bool operator!=(float) const;

        bool any_empty() const;
        std::size_t argmax() const;
        float& back();
        const float& back() const;
        typename std::vector<float>::iterator begin();
        typename std::vector<float>::const_iterator begin() const;
        void emplace_back(const float&);
        void emplace_back(float&&);
        bool empty() const;
        typename std::vector<float>::iterator end();
        typename std::vector<float>::const_iterator end() const;
        float& front();
        const float& front() const;
        bool front_empty() const;
        void push_back(const float&);
        void push_back(float&&);
        void resize(std::size_t);
        void resize(std::size_t, const float&);
        std::size_t size() const;
        std::vector<std::size_t> shape() const;
        std::vector<float>& vector();
        const std::vector<float>& vector() const;

        static Tensor<1,float> copy(const std::vector<float>&);
        template <class U> static Tensor<1,float> copy(const std::vector<U>&);
        static Tensor<1,float> move(std::vector<float>&&);
        template <class U> static Tensor<1,float> move(std::vector<U>&&);
        static float precision();
        static float precision(float);
};

} // type

#include "tensor1f.hpp"