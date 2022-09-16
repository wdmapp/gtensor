#ifndef GTENSOR_DEVICE_BACKEND_H
#define GTENSOR_DEVICE_BACKEND_H

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
#ifdef GTENSOR_HAVE_THRUST
#include "backend_thrust.h"
#endif

// ======================================================================
// backend being used in clib (ie., the Fortran interface)

namespace gt
{
namespace backend
{

template <typename S>
using gallocator = gt::backend::allocator_impl::gallocator<S>;

#if GTENSOR_DEVICE_CUDA
using backend_ops_clib = backend_ops<gt::space::cuda>;
#elif GTENSOR_DEVICE_HIP
using backend_ops_clib = backend_ops<gt::space::hip>;
#elif GTENSOR_DEVICE_SYCL
using backend_ops_clib = backend_ops<gt::space::sycl>;
#else
using backend_ops_clib = backend_ops<gt::space::host>;
#endif

template <typename Ptr>
bool is_device_address(Ptr ptr)
{
  return backend_ops<gt::space::device>::is_device_address(ptr);
}

template <typename Ptr>
inline memory_type get_memory_type(const Ptr ptr)
{
  return backend_ops<gt::space::device>::get_memory_type(ptr);
}

template <typename T>
static void prefetch_device(T* p, size_type n)
{
  return backend_ops<gt::space::device>::prefetch_device(p, n);
}

template <typename T>
static void prefetch_host(T* p, size_type n)
{
  return backend_ops<gt::space::device>::prefetch_host(p, n);
}

} // namespace backend

template <typename T, typename S = gt::space::device>
using device_allocator = typename backend::allocator_impl::selector<T, S>::type;

template <typename T, typename S = gt::space::host>
using host_allocator = typename backend::allocator_impl::selector<T, S>::type;

template <typename T, typename S = gt::space::managed>
using managed_allocator =
  typename backend::allocator_impl::selector<T, S>::type;

// ======================================================================
// fill

template <
  typename Ptr, typename T,
  std::enable_if_t<
    std::is_convertible<T, typename pointer_traits<Ptr>::element_type>::value,
    int> = 0>
void fill(Ptr first, Ptr last, const T& value)
{
  return backend::fill_impl::fill(typename pointer_traits<Ptr>::space_type{},
                                  first, last, value);
}

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
  gt::backend::backend_ops_clib::device_synchronize();
}

} // namespace gt

#endif // GENSOR_DEVICE_BACKEND_H
