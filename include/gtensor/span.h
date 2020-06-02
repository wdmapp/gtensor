
#ifndef GTENSOR_SPAN_H
#define GTENSOR_SPAN_H

#include <cassert>

#include "defs.h"

#ifdef GTENSOR_HAVE_DEVICE
#ifdef GTENSOR_USE_THRUST
#include <thrust/device_ptr.h>
#else
#include "gtensor_storage.h"
#endif
#endif

namespace gt
{

// ======================================================================
// span
//
// very minimal, just enough to support making a gtensor_view

template <typename T>
class span
{
public:
  using value_type = T;
  using pointer = T*;
  using reference = T&;
  using const_pointer = const T*;
  using const_reference = const T&;
  using size_type = gt::size_type;

  span() = default;
  span(pointer data, size_type size) : data_{data}, size_{size} {}

  GT_INLINE const_pointer data() const { return data_; }
  GT_INLINE pointer data() { return data_; }

  GT_INLINE const_reference operator[](size_type i) const
  {
    assert(i < size_);
    return data_[i];
  }
  GT_INLINE reference operator[](size_type i) { return data_[i]; }

private:
  pointer data_ = nullptr;
  size_type size_ = 0;
};

#ifdef GTENSOR_HAVE_DEVICE

// ======================================================================
// device_span
//
// for a gtensor_view of device memory

#ifdef GTENSOR_USE_THRUST

template <typename T>
class device_span
{
public:
  using value_type = T;
  using pointer = thrust::device_ptr<T>;
  using reference = thrust::device_reference<T>;
  using const_pointer = const pointer;
  using const_reference = const reference;
  using size_type = gt::size_type;

  device_span() = default;
  device_span(pointer data, size_type size) : data_{data}, size_{size} {}

  GT_INLINE const_pointer data() const { return data_; }
  GT_INLINE pointer data() { return data_; }

  GT_INLINE const_reference operator[](size_type i) const { return data_[i]; }
  GT_INLINE reference operator[](size_type i) { return data_[i]; }

private:
  pointer data_;
  size_type size_ = 0;
};

#else // not GTENSOR_USE_THRUST

template <typename T>
class device_span
{
public:
  using value_type = T;

  using pointer = typename std::add_pointer<value_type>::type;
  using const_pointer = typename std::add_const<pointer>::type;
  using reference = typename std::add_lvalue_reference<value_type>::type;
  using const_reference = typename std::add_const<reference>::type;
  using size_type = gt::size_type;

  device_span() = default;
  device_span(pointer data, size_type size) : data_{data}, size_{size} {}

  GT_INLINE const_pointer data() const { return data_; }
  GT_INLINE pointer data() { return data_; }

  GT_INLINE const_reference operator[](size_type i) const { return data_[i]; }
  GT_INLINE reference operator[](size_type i) { return data_[i]; }

private:
  pointer data_;
  size_type size_ = 0;
};

#endif // GTENSOR_USE_THRUST

#endif // GTENSOR_HAVE_DEVICE

} // namespace gt

#endif
