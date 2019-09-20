
// ======================================================================
// macros.h
//
// Macros to help with portability with / without CUDA
//
// Copyright (C) 2019 Kai Germaschewski

#ifndef GTENSOR_MACROS_H
#define GTENSOR_MACROS_H

#ifdef __CUDACC__

#define GT_INLINE __host__ __device__ /* FIXME inline */
#define GT_LAMBDA [=] __host__ __device__

#else

#define GT_INLINE inline
#define GT_LAMBDA []

#endif

#ifndef NDEBUG
#define GT_BOUNDSCHECK
#endif

#ifdef __CUDACC__

#define cudaCheck(what) { doCudaCheck(what, __FILE__, __LINE__); }
inline void doCudaCheck(cudaError_t code, const char *file, int line)
{
  if (code != cudaSuccess) {
    fprintf(stderr,"cudaCheck: %d (%s) %s %d\n", code, cudaGetErrorString(code), file, line);
    abort();
  }
}

#define cudaSyncIfEnabled() cudaCheck(cudaDeviceSynchronize())

#endif

#endif
