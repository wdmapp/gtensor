
// ======================================================================
// sarray.h
//
// sarray<T, N> : stack allocated array
//
// Copyright (C) 2019 Kai Germaschewski

#ifndef GTENSOR_SARRAY_H
#define GTENSOR_SARRAY_H

#include "macros.h"

#include <cassert>
#include <cstddef>
#include <iostream>

namespace gt
{

// ======================================================================
// sarray
//
// stack allocated array class, like <array>, but adds support for CUDA
//
// TODO: add missing functionality

template <typename T, std::size_t N>
class sarray
{
public:
  constexpr static std::size_t dimension = N;

  sarray() = default;

  // construct from exactly N elements provided
  template <typename... U, std::enable_if_t<sizeof...(U) == N, int> = 0>
  GT_INLINE sarray(U... args);
  sarray(const T* p, std::size_t n);
  sarray(const T data[N]);

  template <typename O>
  bool operator==(const O& o) const;
  template <typename O>
  bool operator!=(const O& o) const;

  GT_INLINE constexpr static std::size_t size();

  GT_INLINE const T* data() const { return data_; }
  GT_INLINE T* data() { return data_; }

  GT_INLINE const T& operator[](std::size_t i) const;
  GT_INLINE T& operator[](std::size_t i);

  GT_INLINE const T* begin() const;
  GT_INLINE const T* end() const;
  GT_INLINE T* begin();
  GT_INLINE T* end();

private:
  T data_[N] = {};
};

template <typename T, std::size_t N>
inline std::ostream& operator<<(std::ostream& os, const sarray<T, N>& arr);

// zero length array is UB, specialize the 0 dim case used by gscalar
template <typename T>
class sarray<T, 0>
{
public:
  constexpr static std::size_t dimension = 0;

  sarray() = default;

  // all zero dim sarray's are equal, and not equal to nonzero dim
  template <typename O>
  bool operator==(const O& o) const
  {
    return O::dimension == 0;
  }
  template <typename O>
  bool operator!=(const O& o) const
  {
    return O::dimension != 0;
  }

  GT_INLINE constexpr static std::size_t size() { return 0; }

  GT_INLINE const T* data() const { return nullptr; }
  GT_INLINE T* data() { return nullptr; }

  GT_INLINE const T& operator[](T i) const
  {
    gtGpuAssert(false, "zero length sarray does not support element access");
#if defined(GTENSOR_DEVICE_SYCL) || defined(GTENSOR_DEVICE_HIP) ||             \
  (defined(GTENSOR_DEVICE_CUDA) && CUDART_VERSION >= 11300)
    __builtin_unreachable();
#else
    // unreacheable
    static T a = 0;
    return a;
#endif
  }

  GT_INLINE T& operator[](T i)
  {
    gtGpuAssert(false, "zero length sarray does not support element access");
#if defined(GTENSOR_DEVICE_SYCL) || defined(GTENSOR_DEVICE_HIP) ||             \
  (defined(GTENSOR_DEVICE_CUDA) && CUDART_VERSION >= 11300)
    __builtin_unreachable();
#else
    // unreacheable
    static T a = 0;
    return a;
#endif
  }

  GT_INLINE const T* begin() const { return nullptr; }
  GT_INLINE const T* end() const { return nullptr; }
};

// ======================================================================
// sarray implementation

template <typename T, std::size_t N>
template <typename... U, std::enable_if_t<sizeof...(U) == N, int>>
GT_INLINE sarray<T, N>::sarray(U... args) : data_{T(args)...}
{}

template <typename T, std::size_t N>
sarray<T, N>::sarray(const T* p, std::size_t n)
{
  assert(n == N);
  std::copy(p, p + n, data_);
}

template <typename T, std::size_t N>
sarray<T, N>::sarray(const T data[N])
{
  std::copy(data, data + N, data_);
}

template <typename T, std::size_t N>
template <typename O>
inline bool sarray<T, N>::operator==(const O& o) const
{
  if (size() != o.size()) {
    return false;
  }
  return std::equal(begin(), end(), std::begin(o));
}

template <typename T, std::size_t N>
template <typename O>
inline bool sarray<T, N>::operator!=(const O& o) const
{
  return !(*this == o);
}

template <typename T, std::size_t N>
GT_INLINE constexpr std::size_t sarray<T, N>::size()
{
  return N;
}

template <typename T, std::size_t N>
GT_INLINE const T& sarray<T, N>::operator[](std::size_t i) const
{
  return data_[i];
}
template <typename T, std::size_t N>
GT_INLINE T& sarray<T, N>::operator[](std::size_t i)
{
  return data_[i];
}

template <typename T, std::size_t N>
GT_INLINE const T* sarray<T, N>::begin() const
{
  return data_;
}

template <typename T, std::size_t N>
GT_INLINE const T* sarray<T, N>::end() const
{
  return data_ + N;
}

template <typename T, std::size_t N>
GT_INLINE T* sarray<T, N>::begin()
{
  return data_;
}

template <typename T, std::size_t N>
GT_INLINE T* sarray<T, N>::end()
{
  return data_ + N;
}

template <typename T, std::size_t N>
GT_INLINE sarray<T, N + 1> insert(const sarray<T, N>& in, std::size_t i,
                                  T value)
{
  sarray<T, N + 1> out;
  for (int j = 0; j < i; j++) {
    out[j] = in[j];
  }
  out[i] = value;
  for (int j = i; j < N; j++) {
    out[j + 1] = in[j];
  }
  return out;
}

template <typename T, std::size_t N>
GT_INLINE sarray<T, N - 1> remove(const sarray<T, N>& in, std::size_t i)
{
  sarray<T, N - 1> out;
  for (int j = 0; j < i; j++) {
    out[j] = in[j];
  }
  for (int j = i; j < N - 1; j++) {
    out[j] = in[j + 1];
  }
  return out;
}

template <typename T, std::size_t N>
inline std::string to_string(const sarray<T, N>& arr)
{
  std::string s = "{";
  for (std::size_t i = 0; i != N; i++) {
    s += std::to_string(arr[i]);
    if (i != N - 1) {
      s += ", ";
    }
  }
  s += "}";
  return s;
}

template <typename T, std::size_t N>
inline std::ostream& operator<<(std::ostream& os, const sarray<T, N>& arr)
{
  return os << to_string(arr);
}

} // namespace gt

#endif
