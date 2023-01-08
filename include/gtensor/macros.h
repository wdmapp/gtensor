
// ======================================================================
// macros.h
//
// Macros to help with portability with / without GPU and CUDA vs HIP
//
// Copyright (C) 2019 Kai Germaschewski

#ifndef GTENSOR_MACROS_H
#define GTENSOR_MACROS_H

#include <cstdio>

#include "gtensor/device_runtime.h"

#if defined(GTENSOR_DEVICE_CUDA) || defined(GTENSOR_DEVICE_HIP)

#define GT_INLINE inline __host__ __device__
#define GT_LAMBDA [=] __host__ __device__

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define GTENSOR_DEVICE_ONLY
#endif

#elif defined(GTENSOR_DEVICE_SYCL)

#define GT_INLINE inline
#define GT_LAMBDA [=]

#ifdef __SYCL_DEVICE_ONLY__
#define GTENSOR_DEVICE_ONLY
#endif

#else // HOST

#define GT_INLINE inline
#define GT_LAMBDA [=]

#endif

#ifdef GTENSOR_DEVICE_ONLY
#include <cassert>
#else
#include <stdexcept>
#endif

#ifdef GTENSOR_DEVICE_CUDA

#define gtLaunchKernel(kernelName, numblocks, numthreads, memperblock,         \
                       streamId, ...)                                          \
  do {                                                                         \
    kernelName<<<numblocks, numthreads, memperblock, streamId>>>(__VA_ARGS__); \
    gtLaunchCheck(numblocks, numthreads, __FILE__, __LINE__);                  \
  } while (0)
inline void gtLaunchCheck(dim3 numblocks, dim3 numthreads, const char* file,
                          int line)
{
  if (cudaGetLastError() != cudaSuccess) {
    fprintf(stderr, "launch failed %s %d\n", file, line);
    fprintf(stderr, "blocks was [%d, %d, %d]\n", numblocks.x, numblocks.y,
            numblocks.z);
    fprintf(stderr, "threads was [%d, %d, %d]\n", numthreads.x, numthreads.y,
            numthreads.z);
    abort();
  }
}

#define gtGpuCheck(what)                                                       \
  {                                                                            \
    doCudaCheck(what, __FILE__, __LINE__);                                     \
  }
inline void doCudaCheck(cudaError_t code, const char* file, int line)
{
  if (code != cudaSuccess) {
    fprintf(stderr, "gpuCheck: %d (%s) %s %d\n", code, cudaGetErrorString(code),
            file, line);
    abort();
  }
}

inline auto gpuGetLastError() { return cudaGetLastError(); }

#elif defined(GTENSOR_DEVICE_HIP)

#define gtLaunchKernel(kernelName, numblocks, numthreads, ...)                 \
  {                                                                            \
    hipLaunchKernelGGL(kernelName, numblocks, numthreads, __VA_ARGS__);        \
    gtLaunchCheck(numblocks, numthreads, __FILE__, __LINE__);                  \
  }
inline void gtLaunchCheck(dim3 numblocks, dim3 numthreads, const char* file,
                          int line)
{
  if (hipGetLastError() != hipSuccess) {
    fprintf(stderr, "launch failed %s %d\n", file, line);
    fprintf(stderr, "blocks was [%d, %d, %d]\n", numblocks.x, numblocks.y,
            numblocks.z);
    fprintf(stderr, "threads was [%d, %d, %d]\n", numthreads.x, numthreads.y,
            numthreads.z);
    abort();
  }
}

#define gtGpuCheck(what)                                                       \
  {                                                                            \
    doHipCheck(what, __FILE__, __LINE__);                                      \
  }
inline void doHipCheck(hipError_t code, const char* file, int line)
{
  if (code != hipSuccess) {
    fprintf(stderr, "gpuCheck: %d (%s) %s %d\n", code, hipGetErrorString(code),
            file, line);
    abort();
  }
}

inline auto gpuGetLastError() { return hipGetLastError(); }

#elif defined(GTENSOR_DEVICE_SYCL)

#ifdef GTENSOR_SYNC_KERNELS
#define gpuSyncIfEnabledStream(a_stream_view)                                  \
  do {                                                                         \
    a_stream_view.synchronize();                                               \
  } while (0)
#else // not GTENSOR_SYNC_KERNELS
#define gpuSyncIfEnabledStream(a_stream_view)                                  \
  do {                                                                         \
  } while (0)
#endif // GTENSOR_SYNC_KERNELS

#endif // end GTENSOR_DEVICE_SYCL

#if defined(GTENSOR_DEVICE_CUDA) || defined(GTENSOR_DEVICE_HIP)

#ifdef GTENSOR_SYNC_KERNELS
#define gpuSyncIfEnabledStream(a_stream_view)                                  \
  do {                                                                         \
    gtGpuCheck(gpuGetLastError());                                             \
    a_stream_view.synchronize();                                               \
  } while (0)
#else
#define gpuSyncIfEnabledStream(a_stream_view)                                  \
  do {                                                                         \
    gtGpuCheck(gpuGetLastError());                                             \
  } while (0)
#endif

#elif GTENSOR_DEVICE_SYCL

#else

#define gpuSyncIfEnabledStream(a_stream_view)                                  \
  do {                                                                         \
  } while (0)

#endif // end CUDA or HIP

#define gpuSyncIfEnabled() gpuSyncIfEnabledStream(gt::stream_view{})

#ifdef GTENSOR_DEVICE_ONLY
#define gtGpuAssert(cond, errmsg) assert((cond))
#else
#define gtGpuAssert(cond, errmsg)                                              \
  do {                                                                         \
    if (!(cond)) {                                                             \
      throw std::runtime_error(errmsg);                                        \
    }                                                                          \
  } while (0)
#endif

#endif // GTENSORS_MACROS_H
