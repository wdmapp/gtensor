
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

#elif defined(GTENSOR_DEVICE_SYCL)

#define GT_INLINE inline
#define GT_LAMBDA [=]

#else

#define GT_INLINE inline
#define GT_LAMBDA [=]

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

#ifndef NDEBUG
#define gpuSyncIfEnabled()                                                     \
  do {                                                                         \
    gtGpuCheck(cudaGetLastError());                                            \
    gtGpuCheck(cudaDeviceSynchronize());                                       \
  } while (0)
#else
#define gpuSyncIfEnabled()                                                     \
  do {                                                                         \
    gtGpuCheck(cudaGetLastError());                                            \
  } while (0)
#endif

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

#ifndef NDEBUG
#define gpuSyncIfEnabled()                                                     \
  do {                                                                         \
    gtGpuCheck(hipGetLastError());                                             \
    gtGpuCheck(hipDeviceSynchronize());                                        \
  } while (0)
#else // NDEBUG defined
#define gpuSyncIfEnabled()                                                     \
  do {                                                                         \
    gtGpuCheck(hipGetLastError());                                             \
  } while (0)
#endif // NDEBUG

#elif defined(GTENSOR_DEVICE_SYCL)

#ifndef NDEBUG
#define gpuSyncIfEnabled()                                                     \
  do {                                                                         \
    gt::backend::sycl::get_queue().wait();                                     \
  } while (0)
#else // NDEBUG defined
#define gpuSyncIfEnabled()                                                     \
  do {                                                                         \
  } while (0)
#endif // NDEBUG

#endif // end GTENSOR_HAVE_DEVICE

#endif // GTENSORS_MACROS_H
