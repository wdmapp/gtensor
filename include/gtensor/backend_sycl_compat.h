#ifndef GTENSOR_BACKEND_SYCL_COMPAT_H
#define GTENSOR_BACKEND_SYCL_COMPAT_H

// Header to use instead of */sycl.hpp so the sycl backend can work with both
// old and new Intel oneAPI releases

#if (__INTEL_CLANG_COMPILER && __INTEL_CLANG_COMPILER < 20230000)

// older version, use legacy header locations
#include <CL/sycl.hpp>

#ifdef GTENSOR_DEVICE_SYCL_L0
#include "level_zero/ze_api.h"
#include "level_zero/zes_api.h"

#include "CL/sycl/backend/level_zero.hpp"
#endif

#ifdef GTENSOR_DEVICE_SYCL_OPENCL
#include "CL/sycl/backend/opencl.hpp"
#endif

#else // newer verison, use standard 2020 and ext headers

#include <sycl/sycl.hpp>

#ifdef GTENSOR_DEVICE_SYCL_L0
#include "level_zero/ze_api.h"
#include "level_zero/zes_api.h"

#include "sycl/ext/oneapi/backend/level_zero.hpp"
#endif

#ifdef GTENSOR_DEVICE_SYCL_OPENCL
#include "sycl/backend/opencl.hpp"
#endif

#endif // __INTEL_CLANG_COMPILER

#endif // GTENSOR_BACKEND_SYCL_COMPAT_H
