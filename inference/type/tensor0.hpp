#pragma once

namespace type
{

// Tensor<0,T>

template <class T>
Tensor<0,T>::Tensor(const std::vector<T>& data){}

template <class T>
Tensor<0,T>::Tensor(const std::initializer_list<std::size_t>& sizes)
    : Tensor(std::vector<std::size_t>(sizes)) {}

template <class T>
Tensor<0,T>::Tensor(const std::vector<std::size_t>& sizes)
    : Tensor(sizes.begin(), sizes.end()) {}

template <class T>
Tensor<0,T>::Tensor(std::vector<std::size_t>::const_iterator begin, std::vector<std::size_t>::const_iterator end)
{
    if (begin != end) throw std::runtime_error("Tensor<0,T>::Tensor: invalid size dimension");
}

template <class T>
T& Tensor<0,T>::operator[](int i)
{
    return _data;
}

template <class T>
const T& Tensor<0,T>::operator[](int i) const
{
    return _data;
}

template <class T>
bool Tensor<0,T>::operator==(const Tensor<0,T>& a) const
{
    return _data == a._data;
}

template <class T>
bool Tensor<0,T>::operator!=(const Tensor<0,T>& a) const
{
    return _data != a._data;
}

template <class T>
bool Tensor<0,T>::any_empty() const
{
    return empty();
}

template <class T>
T& Tensor<0,T>::back()
{
    return _data;
}

template <class T>
const T& Tensor<0,T>::back() const
{
    return _data;
}

template <class T>
typename std::vector<T>::iterator Tensor<0,T>::begin()
{
    return _data;
}

template <class T>
typename std::vector<T>::const_iterator Tensor<0,T>::begin() const
{
    return _data;
}

template <class T>
bool Tensor<0,T>::empty() const
{
    return false;
}

template <class T>
typename std::vector<T>::iterator Tensor<0,T>::end()
{
    return _data;
}

template <class T>
typename std::vector<T>::const_iterator Tensor<0,T>::end() const
{
    return _data;
}

template <class T>
T& Tensor<0,T>::front()
{
    return _data;
}

template <class T>
const T& Tensor<0,T>::front() const
{
    return _data;
}

template <class T>
bool Tensor<0,T>::front_empty() const
{
    return false;
}

template <class T>
void Tensor<0,T>::resize(std::size_t n) {}

template <class T>
std::size_t Tensor<0,T>::size() const
{
    return 1;
}

template <class T>
std::vector<std::size_t> Tensor<0,T>::shape() const
{
    return {};
}

template <class T>
const std::vector<T>& Tensor<0,T>::vector() const
{
    return {};
}

} // type