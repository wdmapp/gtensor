#ifndef GTENSOR_COMPLEX_OPS_H
#define GTENSOR_COMPLEX_OPS_H

#include <complex>
#include <type_traits>

#include "device_runtime.h"
#include "macros.h"

namespace gt
{

#if defined(GTENSOR_DEVICE_CUDA) || defined(GTENSOR_DEVICE_HIP)

// ======================================================================
// abs

template <typename T>
GT_INLINE T abs(const complex<T>& a)
{
  return thrust::abs(a);
}

template <typename T>
GT_INLINE std::enable_if_t<std::is_floating_point<T>::value, T> abs(const T a)
{
  return gt::abs(complex<T>(a, 0));
}

template <typename T>
GT_INLINE auto abs(const std::complex<T>& a)
{
  return gt::abs(complex<T>(a));
}

template <typename T>
GT_INLINE auto abs(thrust::device_reference<T> a)
{
  return gt::abs(thrust::raw_reference_cast(a));
}

template <typename T>
GT_INLINE auto abs(thrust::device_reference<const T> a)
{
  return gt::abs(thrust::raw_reference_cast(a));
}

// ======================================================================
// conj

using thrust::conj;

template <typename T>
GT_INLINE auto conj(const std::complex<T>& a)
{
  return std::complex<T>(gt::conj(complex<T>(a)));
}

template <typename T>
GT_INLINE auto conj(thrust::device_reference<T> a)
{
  return gt::conj(thrust::raw_reference_cast(a));
}

template <typename T>
GT_INLINE auto conj(thrust::device_reference<const T> a)
{
  return gt::conj(thrust::raw_reference_cast(a));
}

// ======================================================================
// exp

using std::exp;
using thrust::exp;

template <typename T>
GT_INLINE auto exp(const std::complex<T>& a)
{
  return std::complex<T>(gt::exp(complex<T>(a)));
}

template <typename T>
GT_INLINE auto exp(thrust::device_reference<T> a)
{
  return gt::exp(thrust::raw_reference_cast(a));
}

template <typename T>
GT_INLINE auto exp(thrust::device_reference<const T> a)
{
  return gt::exp(thrust::raw_reference_cast(a));
}

// ======================================================================
// norm

using thrust::norm;

template <typename T>
GT_INLINE std::enable_if_t<std::is_floating_point<T>::value, T> norm(const T a)
{
  return a * a;
}

template <typename T>
GT_INLINE auto norm(const std::complex<T>& a)
{
  return gt::norm(complex<T>(a));
}

template <typename T>
GT_INLINE auto norm(thrust::device_reference<T> a)
{
  return gt::norm(thrust::raw_reference_cast(a));
}

template <typename T>
GT_INLINE auto norm(thrust::device_reference<const T> a)
{
  return gt::norm(thrust::raw_reference_cast(a));
}

#elif defined(GTENSOR_DEVICE_SYCL)

using gt::sycl_cplx::abs;
using gt::sycl_cplx::conj;
using gt::sycl_cplx::exp;
using gt::sycl_cplx::norm;

// real version from stdlib
using std::abs;
using std::exp;

#else // host, use std lib

using std::abs;
using std::conj;
using std::exp;
using std::norm;

#endif

} // namespace gt

#endif
