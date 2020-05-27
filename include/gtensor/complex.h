
#ifndef GTENSOR_COMPLEX_H
#define GTENSOR_COMPLEX_H

#ifdef GTENSOR_USE_THRUST
#include <thrust/complex.h>
#else
#include <complex>
#endif

namespace gt
{

#ifdef GTENSOR_USE_THRUST
template <typename T>
using complex = thrust::complex<T>;
#else
template <typename T>
using complex = std::complex<T>;
#endif

} // namespace gt

#endif
