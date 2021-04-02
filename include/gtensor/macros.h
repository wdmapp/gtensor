
// ======================================================================
// macros.h
//
// Macros to help with portability with / without GPU and CUDA vs HIP
//
// Copyright (C) 2019 Kai Germaschewski

#ifndef GTENSOR_MACROS_H
#define GTENSOR_MACROS_H

#include <cstdio>

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

// Note: SYCL doesn't allow variadic function calls in kernel code, so the
// bounds check implementation does not currently work.
#if !defined(NDEBUG) && !defined(GTENSOR_DEVICE_SYCL)
#define GT_BOUNDSCHECK
#endif

#ifdef GTENSOR_DEVICE_CUDA

#define gtLaunchKernel(kernelName, numblocks, numthreads, memperblock,         \
                       streamId, ...)                                          \
  do {                                                                         \
    kernelName<<<numblocks, numthreads, memperblock, streamId>>>(__VA_ARGS__); \
  } while (0)

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

#define gtLaunchKernel(...) hipLaunchKernelGGL(__VA_ARGS__)

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

#endif // end GTENSOR_HAVE_DEVICE

#endif // GTENSORS_MACROS_H
