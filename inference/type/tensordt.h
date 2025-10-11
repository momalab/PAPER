#pragma once

#include <initializer_list>
#include <ostream>
#include <type_traits>
#include <vector>

namespace type
{

template <int D, class T>
class Tensor
{
    private:
        std::vector<Tensor<D-1,T>> _data;

    public:
        Tensor() = default;
        Tensor(const Tensor<D,T>&) = default;
        Tensor(Tensor<D,T>&&) noexcept = default;
        // Tensor(const std::vector<Tensor<D-1,T>>&);
        Tensor(const std::initializer_list<std::size_t>&, const T& = T());
        Tensor(const std::vector<std::size_t>&, const T& = T());
        Tensor(std::vector<std::size_t>::const_iterator, std::vector<std::size_t>::const_iterator, const T& = T());
        ~Tensor() = default;

        Tensor<D,T>& operator=(const Tensor<D,T>&) = default;
        Tensor<D,T>& operator=(Tensor<D,T>&&) noexcept = default;

        Tensor<D-1,T>& operator[](int);
        const Tensor<D-1,T>& operator[](int) const;

        template <class U> Tensor<D,T>& operator+=(const Tensor<D,U>&);
        template <class U> Tensor<D,T>& operator-=(const Tensor<D,U>&);
        template <class U> Tensor<D,T>& operator*=(const Tensor<D,U>&);
        template <class U> Tensor<D,T> operator+(const Tensor<D,U>&) const;
        template <class U> Tensor<D,T> operator-(const Tensor<D,U>&) const;
        template <class U> Tensor<D,T> operator*(const Tensor<D,U>&) const;
        template <class U, typename std::enable_if<!std::is_same<Tensor<D,U>, typename std::decay<U>::type>::value, int>::type = 0> Tensor<D,T>& operator+=(const U&);
        template <class U, typename std::enable_if<!std::is_same<Tensor<D,U>, typename std::decay<U>::type>::value, int>::type = 0> Tensor<D,T>& operator-=(const U&);
        template <class U, typename std::enable_if<!std::is_same<Tensor<D,U>, typename std::decay<U>::type>::value, int>::type = 0> Tensor<D,T>& operator*=(const U&);
        template <class U, typename std::enable_if<!std::is_same<Tensor<D,U>, typename std::decay<U>::type>::value, int>::type = 0> Tensor<D,T> operator+(const U&) const;
        template <class U, typename std::enable_if<!std::is_same<Tensor<D,U>, typename std::decay<U>::type>::value, int>::type = 0> Tensor<D,T> operator-(const U&) const;
        template <class U, typename std::enable_if<!std::is_same<Tensor<D,U>, typename std::decay<U>::type>::value, int>::type = 0> Tensor<D,T> operator*(const U&) const;
        // Tensor<D,T>& operator+=(const T&);
        // Tensor<D,T>& operator-=(const T&);
        // Tensor<D,T>& operator*=(const T&);
        // Tensor<D,T> operator+(const T&) const;
        // Tensor<D,T> operator-(const T&) const;
        // Tensor<D,T> operator*(const T&) const;
        bool operator==(const Tensor<D,T>&) const;
        bool operator!=(const Tensor<D,T>&) const;
        bool operator==(const T&) const;
        bool operator!=(const T&) const;

        bool any_empty() const;
        Tensor<D-1,T>& back();
        const Tensor<D-1,T>& back() const;
        typename std::vector<Tensor<D-1,T>>::iterator begin();
        typename std::vector<Tensor<D-1,T>>::const_iterator begin() const;
        void emplace_back(const Tensor<D-1,T>&);
        void emplace_back(Tensor<D-1,T>&&);
        bool empty() const;
        typename std::vector<Tensor<D-1,T>>::iterator end();
        typename std::vector<Tensor<D-1,T>>::const_iterator end() const;
        Tensor<D-1,T>& front();
        const Tensor<D-1,T>& front() const;
        bool front_empty() const;
        void push_back(const Tensor<D-1,T>&);
        void push_back(Tensor<D-1,T>&&);
        void resize(std::size_t);
        void resize(std::size_t, const Tensor<D-1,T>&);
        std::size_t size() const;
        std::vector<std::size_t> shape() const;
        std::vector<Tensor<D-1,T>>& vector();
        const std::vector<Tensor<D-1,T>>& vector() const;

        static Tensor<D,T> copy(const std::vector<Tensor<D-1,T>>&);
        template <class U> static Tensor<D,T> copy(const std::vector<U>&);
        static Tensor<D,T> move(std::vector<Tensor<D-1,T>>&&);
        template <class U> static Tensor<D,T> move(std::vector<U>&&);
};

} // type