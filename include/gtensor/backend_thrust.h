
#ifndef GTENSOR_BACKEND_THRUST_H
#define GTENSOR_BACKEND_THRUST_H

#include "pointer_traits.h"

#include <thrust/copy.h>
#include <thrust/device_allocator.h>
#include <thrust/fill.h>

// ======================================================================
// gt::backend::host

namespace gt
{
namespace backend
{

// ======================================================================
// backend::thrust

namespace allocator_impl
{
template <typename T>
struct selector<T, gt::space::thrust>
{
#if GTENSOR_DEVICE_CUDA && THRUST_VERSION <= 100903
  using type = ::thrust::device_malloc_allocator<T>;
#else
  using type = ::thrust::device_allocator<T>;
#endif
};
} // namespace allocator_impl

namespace copy_impl
{

template <typename InputPtr, typename OutputPtr>
inline void copy_n(gt::space::thrust tag_in, gt::space::thrust tag_out,
                   InputPtr in, size_type count, OutputPtr out)
{
  ::thrust::copy_n(in, count, out);
}

template <typename InputPtr, typename OutputPtr>
inline void copy_n(gt::space::thrust tag_in, gt::space::host tag_out,
                   InputPtr in, size_type count, OutputPtr out)
{
  ::thrust::copy_n(in, count, out);
}

template <typename InputPtr, typename OutputPtr>
inline void copy_n(gt::space::host tag_in, gt::space::thrust tag_out,
                   InputPtr in, size_type count, OutputPtr out)
{
  ::thrust::copy_n(in, count, out);
}

} // namespace copy_impl

namespace fill_impl
{
template <typename Ptr, typename T>
inline void fill(gt::space::thrust tag, Ptr first, Ptr last, const T& value)
{
  ::thrust::fill(first, last, value);
}
} // namespace fill_impl

} // namespace backend
} // namespace gt

#endif // GTENSOR_BACKEND_THRUST_H
