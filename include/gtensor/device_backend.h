#ifndef GTENSOR_DEVICE_BACKEND_H
#define GTENSOR_DEVICE_BACKEND_H

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <type_traits>

#ifdef GTENSOR_HAVE_DEVICE

#ifdef GTENSOR_USE_THRUST
#include <thrust/device_vector.h>
#endif

#if defined(GTENSOR_DEVICE_CUDA) || defined(GTENSOR_USE_THRUST)
#include "thrust_ext.h"
#endif

#endif // GTENSOR_HAVE_DEVICE

#include "defs.h"
#include "macros.h"
#include "pointer_traits.h"
#include "space_forward.h"

namespace gt
{
namespace space
{

// ======================================================================
// space_traits

template <typename S>
struct space_traits;

template <>
struct space_traits<host>
{
  template <typename T>
  using pointer = T*;
};

#ifdef GTENSOR_HAVE_DEVICE

template <>
struct space_traits<device>
{
#ifdef GTENSOR_USE_THRUST
  template <typename T>
  using pointer = ::thrust::device_ptr<T>;
#else
  template <typename T>
  using pointer = gt::device_ptr<T>;
#endif
};

#endif

} // namespace space

namespace backend
{

template <typename P>
GT_INLINE auto raw_pointer_cast(P p)
{
  return gt::pointer_traits<P>::get(p);
}

template <typename T>
GT_INLINE auto device_pointer_cast(T* p)
{
  using pointer =
    typename gt::space::space_traits<gt::space::device>::template pointer<T>;
  return pointer(p);
}

// ======================================================================

template <typename T, typename A, typename S>
struct wrap_allocator
{
  using value_type = T;
  using pointer = typename gt::space::space_traits<S>::template pointer<T>;
  using size_type = gt::size_type;

  pointer allocate(size_type n) { return pointer(A::template allocate<T>(n)); }
  void deallocate(pointer p, size_type n)
  {
    A::deallocate(gt::pointer_traits<pointer>::get(p));
  }
};

namespace allocator_impl
{
template <typename T, typename S>
struct selector;
}

template <typename T, typename S = gt::space::device>
using device_allocator = typename allocator_impl::selector<T, S>::type;

template <typename T, typename S = gt::space::host>
using host_allocator = typename allocator_impl::selector<T, S>::type;

} // namespace backend
} // namespace gt

#include "backend_host.h"
#ifdef GTENSOR_DEVICE_CUDA
#include "backend_cuda.h"
#endif
#ifdef GTENSOR_DEVICE_HIP
#include "backend_hip.h"
#endif
#ifdef GTENSOR_DEVICE_SYCL
#include "backend_sycl.h"
#endif
#ifdef GTENSOR_USE_THRUST
#include "backend_thrust.h"
#endif

// ======================================================================
// backend being used in clib (ie., the Fortran interface)

namespace gt
{
namespace backend
{
namespace clib
{
#if GTENSOR_DEVICE_CUDA
using namespace backend::cuda;
#elif GTENSOR_DEVICE_HIP
using namespace backend::hip;
#elif GTENSOR_DEVICE_SYCL
using namespace backend::sycl;
#else // just for device_synchronize()
using namespace backend::host;
#endif
} // namespace clib

template <
  typename Ptr, typename T,
  std::enable_if_t<
    std::is_convertible<T, typename pointer_traits<Ptr>::element_type>::value,
    int> = 0>
void fill(Ptr first, Ptr last, const T& value)
{
  return fill_impl::fill(typename pointer_traits<Ptr>::space_type{}, first,
                         last, value);
}

} // namespace backend

// ======================================================================
// copy_n

template <
  typename InputPtr, typename OutputPtr,
  std::enable_if_t<is_allowed_element_type_conversion<
                     typename pointer_traits<OutputPtr>::element_type,
                     typename pointer_traits<InputPtr>::element_type>::value,
                   int> = 0>
inline void copy_n(InputPtr in, gt::size_type count, OutputPtr out)
{
  return gt::backend::copy_impl::copy_n(
    typename pointer_traits<InputPtr>::space_type{},
    typename pointer_traits<OutputPtr>::space_type{}, in, count, out);
}

// ======================================================================
// synchronize

void inline synchronize()
{
  gt::backend::clib::device_synchronize();
}

} // namespace gt

#endif // GENSOR_DEVICE_BACKEND_H
