
#ifndef GT_SPACE_FORWARD_H
#define GT_SPACE_FORWARD_H

#include "defs.h"

namespace gt
{
namespace space
{

struct host
{};
#ifdef GTENSOR_USE_THRUST
struct thrust
{};
#endif
#ifdef GTENSOR_DEVICE_CUDA
struct cuda
{};
#endif
#ifdef GTENSOR_DEVICE_HIP
struct hip
{};
#endif
#ifdef GTENSOR_DEVICE_SYCL
struct sycl
{};
#endif

#ifdef GTENSOR_HAVE_DEVICE

#if GTENSOR_USE_THRUST
using device = thrust;
#elif GTENSOR_DEVICE_CUDA
using device = cuda;
#elif GTENSOR_DEVICE_HIP
using device = hip;
#elif GTENSOR_DEVICE_SYCL
using device = sycl;
#endif

#else // !  GTENSOR_HAVE_DEVICE

using device = host;

#endif

} // namespace space
} // namespace gt

#endif
