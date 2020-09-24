
#ifndef GTENSOR_COMPLEX_H
#define GTENSOR_COMPLEX_H

#if defined(GTENSOR_DEVICE_CUDA) || defined(GTENSOR_USE_THRUST)
#include <thrust/complex.h>
#else
#include <complex>
#endif

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

template <typename T>
struct is_complex : public std::false_type
{};

template <typename T>
struct is_complex<complex<T>> : public std::true_type
{};

} // namespace gt

#endif
