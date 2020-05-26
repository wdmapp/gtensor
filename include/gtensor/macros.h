
// ======================================================================
// macros.h
//
// Macros to help with portability with / without GPU and CUDA vs HIP
//
// Copyright (C) 2019 Kai Germaschewski

#ifndef GTENSOR_MACROS_H
#define GTENSOR_MACROS_H

#if GTENSOR_HAVE_DEVICE

#define GT_INLINE __host__ __device__
#define GT_LAMBDA [=] __host__ __device__

#else

#define GT_INLINE inline
#define GT_LAMBDA []

#endif

#ifndef NDEBUG
#define GT_BOUNDSCHECK
#endif

#ifdef GTENSOR_DEVICE_CUDA

#define gtLaunchKernel(kernelName, numblocks, numthreads, memperblock, streamId, ...)          \
    do {                                                                                           \
        kernelName<<<numblocks, numthreads, memperblock, streamId>>>(__VA_ARGS__);                 \
    } while (0)

#define cudaCheck(what)                                                        \
  {                                                                            \
    doCudaCheck(what, __FILE__, __LINE__);                                     \
  }
inline void doCudaCheck(cudaError_t code, const char* file, int line)
{
  if (code != cudaSuccess) {
    fprintf(stderr, "cudaCheck: %d (%s) %s %d\n", code,
            cudaGetErrorString(code), file, line);
    abort();
  }
}

#ifndef NDEBUG
#define cudaSyncIfEnabled()                                                    \
  do {                                                                         \
    cudaCheck(cudaGetLastError());                                             \
    cudaCheck(cudaDeviceSynchronize());                                        \
  } while (0)
#else
#define cudaSyncIfEnabled()                                                    \
  do {                                                                         \
    cudaCheck(cudaGetLastError());                                             \
  } while (0)
#endif

#elif defined(GTENSOR_DEVICE_HIP)

#define gtLaunchKernel(...)    hipLaunchKernelGGL(__VA_ARGS__)

#define cudaCheck(what)                                                        \
  {                                                                            \
    doHipCheck(what, __FILE__, __LINE__);                                     \
  }
inline void doHipCheck(hipError_t code, const char* file, int line)
{
  if (code != hipSuccess) {
    fprintf(stderr, "gpuCheck: %d (%s) %s %d\n", code,
            hipGetErrorString(code), file, line);
    abort();
  }
}

#ifndef NDEBUG
#define cudaSyncIfEnabled()                                                    \
  do {                                                                         \
    cudaCheck(hipGetLastError());                                             \
    cudaCheck(hipDeviceSynchronize());                                        \
  } while (0)
#else // NDEBUG defined
#define cudaSyncIfEnabled()                                                    \
  do {                                                                         \
    cudaCheck(hipGetLastError());                                             \
  } while (0)
#endif // NDEBUG

#endif // end GTENSOR_HAVE_DEVICE

#endif // GTENSORS_MACROS_H
