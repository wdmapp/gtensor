#ifndef GTENSOR_COMPLEX_OPS_H
#define GTENSOR_COMPLEX_OPS_H

#include "device_runtime.h"
#include "macros.h"

namespace gt
{

#if defined(GTENSOR_DEVICE_CUDA) || defined(GTENSOR_USE_THRUST)

template <typename T>
GT_INLINE T norm(const complex<T>& a)
{
  return thrust::norm(a);
}

template <typename T>
GT_INLINE complex<T> conj(const complex<T>& a)
{
  return thrust::conj(a);
}

template <typename T>
GT_INLINE T norm(thrust::device_reference<complex<T>> a)
{
  return thrust::norm(thrust::raw_reference_cast(a));
}

template <typename T>
GT_INLINE complex<T> conj(thrust::device_reference<complex<T>> a)
{
  return thrust::conj(thrust::raw_reference_cast(a));
}

#else // not CUDA and GTENSOR_USE_THRUST not defined

template <typename T>
GT_INLINE T norm(const complex<T>& a)
{
  return std::norm(a);
}

template <typename T>
GT_INLINE complex<T> conj(const complex<T>& a)
{
  return std::conj(a);
}

#endif

} // namespace gt

#endif
