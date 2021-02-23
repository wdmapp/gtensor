
#ifndef GTENSOR_COMPLEX_H
#define GTENSOR_COMPLEX_H

#if defined(GTENSOR_DEVICE_CUDA) || defined(GTENSOR_USE_THRUST)
#include <thrust/complex.h>
#else
#include <complex>
#endif

#include "gtensor/meta.h"

namespace gt
{

// NOTE: use thrust complex even when using CUDA with the internal
// gtensor_storage, otherwise the complex operators are missing and the
// header should be present in all cuda versions.
#if defined(GTENSOR_DEVICE_CUDA) || defined(GTENSOR_USE_THRUST)

template <typename T>
using complex = thrust::complex<T>;

#else // not CUDA and GTENSOR_USE_THRUST not defined

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
