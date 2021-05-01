
#ifndef POINTER_TRAITS_H
#define POINTER_TRAITS_H

#include "device_ptr.h"
#ifdef GTENSOR_USE_THRUST
#include <thrust/device_ptr.h>
#endif

namespace gt
{

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

  static pointer get(pointer p) { return p; }
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

  static T* get(pointer p) { return p.get(); }
};

#endif

} // namespace gt

#endif