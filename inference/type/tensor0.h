#pragma once

namespace type
{

template <class T>
class Tensor<0,T>
{
    private:
        T _data;

    public:
        Tensor() = default;
        Tensor(const Tensor<0,T>&) = default;
        Tensor(Tensor<0,T>&&) = default;
        Tensor(const std::vector<T>&);
        Tensor(const std::initializer_list<std::size_t>&);
        Tensor(const std::vector<std::size_t>&);
        Tensor(std::vector<std::size_t>::const_iterator, std::vector<std::size_t>::const_iterator);

        Tensor<0,T>& operator=(const Tensor<0,T>&) = default;
        Tensor<0,T>& operator=(Tensor<0,T>&&) = default;

        T& operator[](int);
        const T& operator[](int) const;

        bool operator==(const Tensor<0,T>& a) const;
        bool operator!=(const Tensor<0,T>& a) const;

        bool any_empty() const;
        T& back();
        const T& back() const;
        typename std::vector<T>::iterator begin();
        typename std::vector<T>::const_iterator begin() const;
        bool empty() const;
        typename std::vector<T>::iterator end();
        typename std::vector<T>::const_iterator end() const;
        T& front();
        const T& front() const;
        bool front_empty() const;
        void resize(std::size_t);
        std::size_t size() const;
        std::vector<std::size_t> shape() const;
        const std::vector<T>& vector() const;
};

} // type