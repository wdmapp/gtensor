
#ifndef POINTER_TRAITS_H
#define POINTER_TRAITS_H

#include "device_ptr.h"
#include "space_forward.h"

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
  using space_type = gt::space::host;

  template <typename U>
  using rebind = U*;

  GT_INLINE static pointer get(pointer p) { return p; }
};

template <typename T>
struct pointer_traits<gt::backend::device_ptr<T>>
{
  using element_type = T;
  using pointer = gt::backend::device_ptr<T>;
  using const_pointer = gt::backend::device_ptr<const T>;
  using reference = T&;
  using const_reference = const T&;
  using space_type = typename pointer::space_type;

  template <typename U>
  using rebind = gt::backend::device_ptr<U>;

  GT_INLINE static T* get(pointer p) { return p.get(); }
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
  using space_type = gt::space::thrust;

  template <typename U>
  using rebind = ::thrust::device_ptr<U>;

  GT_INLINE static T* get(pointer p) { return p.get(); }
};

#endif

} // namespace gt

#endif