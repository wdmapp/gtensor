#ifndef GTENSOR_BLAS_CUDA_H
#define GTENSOR_BLAS_CUDA_H

#include "cublas_v2.h"

typedef cublasHandle_t gpublas_handle_t;
typedef cudaStream_t gpublas_stream_t;
typedef cuDoubleComplex gpublas_complex_double_t;
typedef cuComplex gpublas_complex_float_t;
typedef int gpublas_index_t;

namespace gt
{

namespace blas
{

// ======================================================================
// types aliases

using handle_t = cublasHandle_t;
using stream_t = cudaStream_t;
using index_t = int;

// ======================================================================
// handle and stream management

inline void create(handle_t* handle)
{
  gtGpuCheck((cudaError_t)cublasCreate(handle));
}

inline void destroy(handle_t handle)
{
  gtGpuCheck((cudaError_t)cublasDestroy(handle));
}

inline void set_stream(handle_t handle, stream_t stream_id)
{
  gtGpuCheck((cudaError_t)cublasSetStream(handle, stream_id));
}

inline void get_stream(handle_t handle, stream_t* stream_id)
{
  gtGpuCheck((cudaError_t)cublasGetStream(handle, stream_id));
}

// ======================================================================
// axpy

template <typename T>
inline void axpy(handle_t h, int n, const T* a, const T* x, int incx, T* y,
                 int incy);

#define CREATE_AXPY(METHOD, GTTYPE, BLASTYPE)                                  \
  template <>                                                                  \
  inline void axpy<GTTYPE>(handle_t h, int n, const GTTYPE* a,                 \
                           const GTTYPE* x, int incx, GTTYPE* y, int incy)     \
  {                                                                            \
    gtGpuCheck((cudaError_t)METHOD(h, n, (BLASTYPE*)a, (BLASTYPE*)x, incx,     \
                                   (BLASTYPE*)y, incy));                       \
  }

CREATE_AXPY(cublasZaxpy, gt::complex<double>, cuDoubleComplex)
CREATE_AXPY(cublasCaxpy, gt::complex<float>, cuComplex)
CREATE_AXPY(cublasDaxpy, double, double)
CREATE_AXPY(cublasSaxpy, float, float)

// ======================================================================
// scal

template <typename T>
inline void scal(handle_t h, int n, const T fac, T* arr, const int incx);

#define CREATE_SCAL(METHOD, GTTYPE, BLASTYPE)                                  \
  template <>                                                                  \
  inline void scal<GTTYPE>(handle_t h, int n, const GTTYPE fac, GTTYPE* arr,   \
                           const int incx)                                     \
  {                                                                            \
    gtGpuCheck(                                                                \
      (cudaError_t)METHOD(h, n, (BLASTYPE*)&fac, (BLASTYPE*)arr, incx));       \
  }

CREATE_SCAL(cublasZscal, gt::complex<double>, cuDoubleComplex)
CREATE_SCAL(cublasCscal, gt::complex<float>, cuComplex)
CREATE_SCAL(cublasDscal, double, double)
CREATE_SCAL(cublasSscal, float, float)

// ======================================================================
// copy

template <typename T>
inline void copy(handle_t h, int n, const T* x, int incx, T* y, int incy);

#define CREATE_COPY(METHOD, GTTYPE, BLASTYPE)                                  \
  template <>                                                                  \
  inline void copy<GTTYPE>(handle_t h, int n, const GTTYPE* x, int incx,       \
                           GTTYPE* y, int incy)                                \
  {                                                                            \
    gtGpuCheck(                                                                \
      (cudaError_t)METHOD(h, n, (BLASTYPE*)x, incx, (BLASTYPE*)y, incy));      \
  }

CREATE_COPY(cublasZcopy, gt::complex<double>, cuDoubleComplex)
CREATE_COPY(cublasCcopy, gt::complex<float>, cuComplex)
CREATE_COPY(cublasDcopy, double, double)
CREATE_COPY(cublasScopy, float, float)

// ======================================================================
// gemv

template <typename T>
inline void gemv(handle_t h, int m, int n, const T* alpha, const T* A, int lda,
                 const T* x, int incx, const T* beta, T* y, int incy);

#define CREATE_GEMV(METHOD, GTTYPE, BLASTYPE)                                  \
  template <>                                                                  \
  inline void gemv<GTTYPE>(handle_t h, int m, int n, const GTTYPE* alpha,      \
                           const GTTYPE* A, int lda, const GTTYPE* x,          \
                           int incx, const GTTYPE* beta, GTTYPE* y, int incy)  \
  {                                                                            \
    gtGpuCheck((cudaError_t)METHOD(h, CUBLAS_OP_N, m, n, (BLASTYPE*)alpha,     \
                                   (BLASTYPE*)A, lda, (BLASTYPE*)x, incx,      \
                                   (BLASTYPE*)beta, (BLASTYPE*)y, incy));      \
  }

CREATE_GEMV(cublasZgemv, gt::complex<double>, cuDoubleComplex)
CREATE_GEMV(cublasCgemv, gt::complex<float>, cuComplex)
CREATE_GEMV(cublasDgemv, double, double)
CREATE_GEMV(cublasSgemv, float, float)

} // end namespace blas

} // end namespace gt

#endif
