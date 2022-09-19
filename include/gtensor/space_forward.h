
#ifndef GT_SPACE_FORWARD_H
#define GT_SPACE_FORWARD_H

#include "defs.h"

namespace gt
{
namespace space
{

struct host_only
{};

#ifdef GTENSOR_HAVE_THRUST
struct thrust
{};
struct thrust_managed
{};
struct thrust_host
{};
#endif

#ifdef GTENSOR_DEVICE_CUDA
struct cuda
{};
struct cuda_managed
{};
struct cuda_host
{};
#endif

#ifdef GTENSOR_DEVICE_HIP
struct hip
{};
struct hip_managed
{};
struct hip_host
{};
#endif

#ifdef GTENSOR_DEVICE_SYCL
struct sycl
{};
struct sycl_managed
{};
struct sycl_host
{};
#endif

#ifdef GTENSOR_HAVE_DEVICE

#if GTENSOR_USE_THRUST
using device = thrust;
using host = thrust_host;
using managed = thrust_managed;
#elif GTENSOR_DEVICE_CUDA
using device = cuda;
using host = cuda_host;
using managed = cuda_managed;
#elif GTENSOR_DEVICE_HIP
using device = hip;
using host = hip_host;
using managed = hip_managed;
#elif GTENSOR_DEVICE_SYCL
using device = sycl;
using host = sycl_host;
using managed = sycl_managed;
#endif

#if GTENSOR_DEVICE_CUDA
using clib_device = cuda;
using clib_host = cuda_host;
using clib_managed = cuda_managed;
#elif GTENSOR_DEVICE_HIP
using clib_device = hip;
using clib_host = hip_host;
using clib_managed = hip_managed;
#elif GTENSOR_DEVICE_SYCL
using clib_device = sycl;
using clib_host = sycl_host;
using clib_managed = sycl_managed;
#endif

#else // !  GTENSOR_HAVE_DEVICE

using host = host_only;
using device = host;
using managed = host;
using clib_host = host;
using clib_device = host;
using clib_managed = host;

#endif

} // namespace space
} // namespace gt

#endif
