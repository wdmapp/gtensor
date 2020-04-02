
#ifndef GTENSOR_COMPLEX_H
#define GTENSOR_COMPLEX_H

#if (__CUDACC__ || __HCC__)
#include <thrust/complex.h>
#else
#include <complex>
#endif

namespace gt
{

#if (__CUDACC__ || __HCC__)
template <typename T>
using complex = thrust::complex<T>;
#else
template <typename T>
using complex = std::complex<T>;
#endif

} // namespace gt

#endif
