
#ifndef GTENSOR_SPAN_H
#define GTENSOR_SPAN_H

#include <cassert>

#if __cplusplus >= 202000L || (defined(_MSVC_LANG) && _MSVC_LANG >= 202000L)
#include <span>
#endif

#include "defs.h"
#include "macros.h"

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
// very minimal, just enough to support making a gtensor_span
// Note that the span has pointer semantics, in that coyping does
// not copy the underlying data, just the pointer and size, and
// requesting access to the underlying data from a const instance
// via data() and operator[] returns a non-const reference allowing
// modification. This is consistent with the C++20 standardized
// span and with gsl::span. To not allow modification, the underlying
// data type can be const.

#if __cplusplus >= 202000L || (defined(_MSVC_LANG) && _MSVC_LANG >= 202000L)

template <typename T>
using span = std::span<T>;

#else // not C++ 20, define subset of span we care about

template <typename P>
struct pointer_traits;

template <typename T>
struct pointer_traits<T*>
{
  using element_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;

  template <typename U>
  using rebind = U*;
};

#ifdef GTENSOR_USE_THRUST

template <typename T>
struct pointer_traits<::thrust::device_ptr<T>>
{
  using element_type = T;
  using pointer = ::thrust::device_ptr<T>;
  using const_pointer = ::thrust::device_ptr<const T>;
  using reference = ::thrust::device_reference<T>;
  using const_reference = ::thrust::device_reference<const T>;

  template <typename U>
  using rebind = ::thrust::device_ptr<U>;
};

#endif

template <typename T, typename Ptr = T*>
class span
{
public:
  using element_type = T;
  using value_type = std::remove_cv_t<T>;

  using pointer = typename pointer_traits<Ptr>::pointer;
  using const_pointer = typename pointer_traits<Ptr>::const_pointer;
  using reference = typename pointer_traits<Ptr>::reference;
  using const_reference = typename pointer_traits<Ptr>::const_reference;
  using iterator = pointer;
  using size_type = gt::size_type;

  span() = default;
  GT_INLINE span(pointer data, size_type size) : data_{data}, size_{size} {}

  span(const span& other) = default;

  template <class OtherT,
            std::enable_if_t<
              is_allowed_element_type_conversion<OtherT, T>::value, int> = 0>
  GT_INLINE span(
    const span<OtherT, typename pointer_traits<Ptr>::template rebind<OtherT>>&
      other)
    : data_{other.data()}, size_{other.size()}
  {}

  span& operator=(const span& other) = default;

  GT_INLINE pointer data() const { return data_; }
  GT_INLINE size_type size() const { return size_; }

  GT_INLINE iterator begin() const { return data_; }
  GT_INLINE iterator end() const { return data_ + size_; }

  GT_INLINE reference operator[](size_type i) const { return data_[i]; }

private:
  pointer data_ = nullptr;
  size_type size_ = 0;
};

#endif // C++20

#ifdef GTENSOR_HAVE_DEVICE

// ======================================================================
// device_span
//
// for a gtensor_span of device memory

#ifdef GTENSOR_USE_THRUST

template <typename T>
using device_span = span<T, thrust::device_ptr<T>>;

#else // not GTENSOR_USE_THRUST

template <typename T>
using device_span = span<T>;

#endif // GTENSOR_USE_THRUST

#endif // GTENSOR_HAVE_DEVICE

} // namespace gt

#endif // GTENSOR_SPAN_H
