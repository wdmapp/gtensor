
#ifndef GTENSOR_COMPLEX_H
#define GTENSOR_COMPLEX_H

#ifdef __CUDACC__
#include <thrust/complex.h>
#else
#include <complex>
#endif

namespace gt
{

#ifdef __CUDACC__
template <typename T>
using complex = thrust::complex<T>;
#else
template <typename T>
using complex = std::complex<T>;
#endif

} // namespace gt

#endif
