
#ifndef GTENSOR_SPACE_H
#define GTENSOR_SPACE_H

#include "defs.h"
#include "span.h"

#include <vector>

#ifdef __CUDACC__
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#endif

namespace gt
{

// ======================================================================
// space

namespace space
{

struct any;

struct kernel;

struct host
{
  template <typename T>
#ifdef __CUDACC__
  using Vector = thrust::host_vector<T>;
#else
  using Vector = std::vector<T>;
#endif
  template <typename T>
  using Span = span<T>;
};

#ifdef __CUDACC__

struct device
{
  template <typename T>
  using Vector = thrust::device_vector<T>;
  template <typename T>
  using Span = device_span<T>;
};

#else

using device = host;

#endif

} // namespace space

} // namespace gt

#endif
