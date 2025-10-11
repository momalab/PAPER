#pragma once

#include <initializer_list>
#include <ostream>
#include <type_traits>
#include <vector>
#include "tensordt.h"

namespace type
{

template <class T>
class Tensor<1,T>
{
    private:
        std::vector<T> _data;

    public:
        Tensor() = default;
        Tensor(const Tensor<1,T>&) = default;
        Tensor(Tensor<1,T>&&) noexcept = default;
        // Tensor(const std::vector<T>&);
        Tensor(const std::initializer_list<std::size_t>&, const T& = T());
        Tensor(const std::vector<std::size_t>&, const T& = T());
        Tensor(std::vector<std::size_t>::const_iterator, std::vector<std::size_t>::const_iterator, const T& = T());
        ~Tensor() = default;

        Tensor<1,T>& operator=(const Tensor<1,T>&) = default;
        Tensor<1,T>& operator=(Tensor<1,T>&&) noexcept = default;

        T& operator[](int i);
        const T& operator[](int i) const;

        template <class U> Tensor<1,T>& operator+=(const Tensor<1,U>&);
        template <class U> Tensor<1,T>& operator-=(const Tensor<1,U>&);
        template <class U> Tensor<1,T>& operator*=(const Tensor<1,U>&);
        template <class U> Tensor<1,T> operator+(const Tensor<1,U>&) const;
        template <class U> Tensor<1,T> operator-(const Tensor<1,U>&) const;
        template <class U> Tensor<1,T> operator*(const Tensor<1,U>&) const;
        template <class U, typename std::enable_if<!std::is_same<Tensor<1,U>, typename std::decay<U>::type>::value, int>::type = 0> Tensor<1,T>& operator+=(const U&);
        template <class U, typename std::enable_if<!std::is_same<Tensor<1,U>, typename std::decay<U>::type>::value, int>::type = 0> Tensor<1,T>& operator-=(const U&);
        template <class U, typename std::enable_if<!std::is_same<Tensor<1,U>, typename std::decay<U>::type>::value, int>::type = 0> Tensor<1,T>& operator*=(const U&);
        template <class U, typename std::enable_if<!std::is_same<Tensor<1,U>, typename std::decay<U>::type>::value, int>::type = 0> Tensor<1,T> operator+(const U&) const;
        template <class U, typename std::enable_if<!std::is_same<Tensor<1,U>, typename std::decay<U>::type>::value, int>::type = 0> Tensor<1,T> operator-(const U&) const;
        template <class U, typename std::enable_if<!std::is_same<Tensor<1,U>, typename std::decay<U>::type>::value, int>::type = 0> Tensor<1,T> operator*(const U&) const;
        bool operator==(const Tensor<1,T>&) const;
        bool operator!=(const Tensor<1,T>&) const;
        bool operator==(const T&) const;
        bool operator!=(const T&) const;

        bool any_empty() const;
        T& back();
        const T& back() const;
        typename std::vector<T>::iterator begin();
        typename std::vector<T>::const_iterator begin() const;
        void emplace_back(const T&);
        void emplace_back(T&&);
        bool empty() const;
        typename std::vector<T>::iterator end();
        typename std::vector<T>::const_iterator end() const;
        T& front();
        const T& front() const;
        bool front_empty() const;
        void push_back(const T&);
        void push_back(T&&);
        void resize(std::size_t);
        void resize(std::size_t, const T&);
        std::size_t size() const;
        std::vector<std::size_t> shape() const;
        std::vector<T>& vector();
        const std::vector<T>& vector() const;

        static Tensor<1,T> copy(const std::vector<T>&);
        template <class U> static Tensor<1,T> copy(const std::vector<U>&);
        static Tensor<1,T> move(std::vector<T>&&);
        template <class U> static Tensor<1,T> move(std::vector<U>&&);
};

} // type