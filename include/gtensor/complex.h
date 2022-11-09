
#ifndef GTENSOR_COMPLEX_H
#define GTENSOR_COMPLEX_H

#if defined(GTENSOR_DEVICE_CUDA) || defined(GTENSOR_DEVICE_HIP)
#include <thrust/complex.h>
#elif defined(GTENSOR_DEVICE_SYCL)
#include "sycl_ext_complex.hpp"
#else
#include <complex>
#endif

#include "gtensor/meta.h"

namespace gt
{

// NOTE: always use thrust complex for CUDA and HIP, regardless of storage
// backend. Depending on ROCm and CUDA verison, using std::complex could
// cause device versions of operators to not be defined properly.
#if defined(GTENSOR_DEVICE_CUDA) || defined(GTENSOR_DEVICE_HIP)

template <typename T>
using complex = thrust::complex<T>;

#elif defined(GTENSOR_DEVICE_SYCL)

// TODO: this will hopefully be standardized soon and be sycl::complex
template <typename T>
using complex = ::sycl::ext::cplx::complex<T>;

#else // fallback to std::complex, e.g. for host backend

template <typename T>
using complex = std::complex<T>;

#endif

// ======================================================================
// is_complex

template <typename T>
struct is_complex : public std::false_type
{};

template <typename T>
struct is_complex<complex<T>> : public std::true_type
{};

template <typename T>
constexpr bool is_complex_v = is_complex<T>::value;

// ======================================================================
// has_complex_value_type

template <typename C, typename = void>
struct has_complex_value_type : std::false_type
{};

template <typename C>
struct has_complex_value_type<
  C, gt::meta::void_t<std::enable_if_t<is_complex_v<typename C::value_type>>>>
  : std::true_type
{};

template <typename T>
constexpr bool has_complex_value_type_v = has_complex_value_type<T>::value;

// ======================================================================
// complex_subtype

template <typename T>
struct complex_subtype
{
  using type = T;
};

template <typename R>
struct complex_subtype<gt::complex<R>>
{
  using type = R;
};

template <typename T>
using complex_subtype_t = typename complex_subtype<T>::type;

// ======================================================================
// container_complex_subtype

template <typename C>
struct container_complex_subtype
{
  using type = complex_subtype_t<typename C::value_type>;
};

template <typename T>
using container_complex_subtype_t = typename container_complex_subtype<T>::type;

} // namespace gt

#endif
