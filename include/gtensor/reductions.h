#ifndef GTENSOR_REDUCTIONS_H
#define GTENSOR_REDUCTIONS_H

#include "defs.h"
#include "device_backend.h"
#include "device_copy.h"

#include "complex.h"
#include "complex_ops.h"
#include "gcontainer.h"
#include "gfunction.h"
#include "gtensor_span.h"
#include "gview.h"
#include "operator.h"

#ifdef GTENSOR_USE_THRUST

namespace gt
{

template <typename T, int N, typename S>
inline T sum(const gtensor_span<const T, N, S>& a)
{
  auto start = a.data();
  auto end = a.data() + a.size();
  return thrust::reduce(start, end, 0., thrust::plus<T>());
}

template <typename T, int N, typename S>
inline T sum(const gtensor<T, N, S>& a)
{
  auto aspan = a.to_kernel();
  return sum(aspan);
}

template <typename T, int N, typename S>
inline T max(const gtensor_span<const T, N, S>& a)
{
  auto start = a.data();
  auto end = a.data() + a.size();
  return thrust::reduce(start, end, 0., thrust::maximum<T>());
}

template <typename T, int N, typename S>
inline T max(const gtensor<T, N, S>& a)
{
  auto aspan = a.to_kernel();
  return max(aspan);
}

template <typename T, int N, typename S>
inline T min(const gtensor_span<const T, N, S>& a)
{
  auto start = a.data();
  auto end = a.data() + a.size();
  auto min_element = thrust::min_element(start, end);
  return *min_element;
}

template <typename T, int N, typename S>
inline T min(const gtensor<T, N, S>& a)
{
  auto aspan = a.to_kernel();
  return min(aspan);
}

} // namespace gt

#endif // GTENSOR_USE_THRUST

#endif // GTENSOR_REDUCTIONS_H
