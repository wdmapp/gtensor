#ifndef GTENSOR_BLAS_HIP_H
#define GTENSOR_BLAS_HIP_H

#include "rocblas.h"
#include "rocsolver.h"

namespace gt
{

namespace blas
{

struct handle_t
{
  rocblas_handle handle;
};

// ======================================================================
// types aliases

using stream_t = hipStream_t;
using index_t = int;

// ======================================================================
// handle and stream management

inline handle_t* create()
{
  handle_t* h = new handle_t();
  gtGpuCheck((hipError_t)rocblas_create_handle(&(h->handle)));
  return h;
}

inline void destroy(handle_t* h)
{
  gtGpuCheck((hipError_t)rocblas_destroy_handle(h->handle));
  delete h;
}

inline void set_stream(handle_t* h, stream_t stream_id)
{
  gtGpuCheck((hipError_t)rocblas_set_stream(h->handle, stream_id));
}

inline void get_stream(handle_t* h, stream_t* stream_id)
{
  gtGpuCheck((hipError_t)rocblas_get_stream(h->handle, stream_id));
}

// ======================================================================
// axpy

template <typename T>
inline void axpy(handle_t* h, int n, T a, const T* x, int incx, T* y, int incy);

#define CREATE_AXPY(METHOD, GTTYPE, BLASTYPE)                                  \
  template <>                                                                  \
  inline void axpy<GTTYPE>(handle_t * h, int n, GTTYPE a, const GTTYPE* x,     \
                           int incx, GTTYPE* y, int incy)                      \
  {                                                                            \
    gtGpuCheck((hipError_t)METHOD(h->handle, n,                                \
                                  reinterpret_cast<BLASTYPE*>(&a),             \
                                  reinterpret_cast<const BLASTYPE*>(x), incx,  \
                                  reinterpret_cast<BLASTYPE*>(y), incy));      \
  }

CREATE_AXPY(rocblas_zaxpy, gt::complex<double>, rocblas_double_complex)
CREATE_AXPY(rocblas_caxpy, gt::complex<float>, rocblas_float_complex)
CREATE_AXPY(rocblas_daxpy, double, double)
CREATE_AXPY(rocblas_saxpy, float, float)

#undef CREATE_AXPY

// ======================================================================
// scal

template <typename S, typename T>
inline void scal(handle_t* h, int n, S fac, T* arr, const int incx);

#define CREATE_SCAL(METHOD, GTTYPE, BLASTYPE)                                  \
  template <>                                                                  \
  inline void scal<GTTYPE, GTTYPE>(handle_t * h, int n, GTTYPE fac,            \
                                   GTTYPE* arr, const int incx)                \
  {                                                                            \
    gtGpuCheck((hipError_t)METHOD(h->handle, n,                                \
                                  reinterpret_cast<BLASTYPE*>(&fac),           \
                                  reinterpret_cast<BLASTYPE*>(arr), incx));    \
  }

CREATE_SCAL(rocblas_zscal, gt::complex<double>, rocblas_double_complex)
CREATE_SCAL(rocblas_cscal, gt::complex<float>, rocblas_float_complex)
CREATE_SCAL(rocblas_dscal, double, double)
CREATE_SCAL(rocblas_sscal, float, float)

#undef CREATE_SCAL

// ======================================================================
// (zd|cs)scal

template <>
inline void scal<double, gt::complex<double>>(handle_t* h, int n, double fac,
                                              gt::complex<double>* arr,
                                              const int incx)
{
  gtGpuCheck((hipError_t)rocblas_zdscal(
    h->handle, n, &fac, reinterpret_cast<rocblas_double_complex*>(arr), incx));
}

template <>
inline void scal<float, gt::complex<float>>(handle_t* h, int n, float fac,
                                            gt::complex<float>* arr,
                                            const int incx)
{
  gtGpuCheck((hipError_t)rocblas_csscal(
    h->handle, n, &fac, reinterpret_cast<rocblas_float_complex*>(arr), incx));
}

// ======================================================================
// copy

template <typename T>
inline void copy(handle_t* h, int n, const T* x, int incx, T* y, int incy);

#define CREATE_COPY(METHOD, GTTYPE, BLASTYPE)                                  \
  template <>                                                                  \
  inline void copy<GTTYPE>(handle_t * h, int n, const GTTYPE* x, int incx,     \
                           GTTYPE* y, int incy)                                \
  {                                                                            \
    gtGpuCheck((hipError_t)METHOD(h->handle, n,                                \
                                  reinterpret_cast<const BLASTYPE*>(x), incx,  \
                                  reinterpret_cast<BLASTYPE*>(y), incy));      \
  }

CREATE_COPY(rocblas_zcopy, gt::complex<double>, rocblas_double_complex)
CREATE_COPY(rocblas_ccopy, gt::complex<float>, rocblas_float_complex)
CREATE_COPY(rocblas_dcopy, double, double)
CREATE_COPY(rocblas_scopy, float, float)

#undef CREATE_COPY

// ======================================================================
// dot, dotc (conjugate)

template <typename T>
inline T dot(handle_t* h, int n, const T* x, int incx, const T* y, int incy);

#define CREATE_DOT(METHOD, GTTYPE, BLASTYPE)                                   \
  template <>                                                                  \
  inline GTTYPE dot<GTTYPE>(handle_t * h, int n, const GTTYPE* x, int incx,    \
                            const GTTYPE* y, int incy)                         \
  {                                                                            \
    GTTYPE result;                                                             \
    gtGpuCheck((hipError_t)METHOD(h->handle, n,                                \
                                  reinterpret_cast<const BLASTYPE*>(x), incx,  \
                                  reinterpret_cast<const BLASTYPE*>(y), incy,  \
                                  reinterpret_cast<BLASTYPE*>(&result)));      \
    return result;                                                             \
  }

CREATE_DOT(rocblas_ddot, double, double)
CREATE_DOT(rocblas_sdot, float, float)

#undef CREATE_DOT

template <typename T>
inline T dotu(handle_t* h, int n, const T* x, int incx, const T* y, int incy);

#define CREATE_DOTU(METHOD, GTTYPE, BLASTYPE)                                  \
  template <>                                                                  \
  inline GTTYPE dotu<GTTYPE>(handle_t * h, int n, const GTTYPE* x, int incx,   \
                             const GTTYPE* y, int incy)                        \
  {                                                                            \
    GTTYPE result;                                                             \
    gtGpuCheck((hipError_t)METHOD(h->handle, n,                                \
                                  reinterpret_cast<const BLASTYPE*>(x), incx,  \
                                  reinterpret_cast<const BLASTYPE*>(y), incy,  \
                                  reinterpret_cast<BLASTYPE*>(&result)));      \
    return result;                                                             \
  }

CREATE_DOTU(rocblas_zdotu, gt::complex<double>, rocblas_double_complex)
CREATE_DOTU(rocblas_cdotu, gt::complex<float>, rocblas_float_complex)

#undef CREATE_DOTU

template <typename T>
inline T dotc(handle_t* h, int n, const T* x, int incx, const T* y, int incy);

#define CREATE_DOTC(METHOD, GTTYPE, BLASTYPE)                                  \
  template <>                                                                  \
  inline GTTYPE dotc<GTTYPE>(handle_t * h, int n, const GTTYPE* x, int incx,   \
                             const GTTYPE* y, int incy)                        \
  {                                                                            \
    GTTYPE result;                                                             \
    gtGpuCheck((hipError_t)METHOD(h->handle, n,                                \
                                  reinterpret_cast<const BLASTYPE*>(x), incx,  \
                                  reinterpret_cast<const BLASTYPE*>(y), incy,  \
                                  reinterpret_cast<BLASTYPE*>(&result)));      \
    return result;                                                             \
  }

CREATE_DOTC(rocblas_zdotc, gt::complex<double>, rocblas_double_complex)
CREATE_DOTC(rocblas_cdotc, gt::complex<float>, rocblas_float_complex)

#undef CREATE_DOTC

// ======================================================================
// gemv

template <typename T>
inline void gemv(handle_t* h, int m, int n, T alpha, const T* A, int lda,
                 const T* x, int incx, T beta, T* y, int incy);

#define CREATE_GEMV(METHOD, GTTYPE, BLASTYPE)                                  \
  template <>                                                                  \
  inline void gemv<GTTYPE>(handle_t * h, int m, int n, GTTYPE alpha,           \
                           const GTTYPE* A, int lda, const GTTYPE* x,          \
                           int incx, GTTYPE beta, GTTYPE* y, int incy)         \
  {                                                                            \
    gtGpuCheck((hipError_t)METHOD(h->handle, rocblas_operation_none, m, n,     \
                                  reinterpret_cast<BLASTYPE*>(&alpha),         \
                                  reinterpret_cast<const BLASTYPE*>(A), lda,   \
                                  reinterpret_cast<const BLASTYPE*>(x), incx,  \
                                  reinterpret_cast<BLASTYPE*>(&beta),          \
                                  reinterpret_cast<BLASTYPE*>(y), incy));      \
  }

CREATE_GEMV(rocblas_zgemv, gt::complex<double>, rocblas_double_complex)
CREATE_GEMV(rocblas_cgemv, gt::complex<float>, rocblas_float_complex)
CREATE_GEMV(rocblas_dgemv, double, double)
CREATE_GEMV(rocblas_sgemv, float, float)

#undef CREATE_GEMV

// ======================================================================
// getrf/getrs batched

template <typename T>
inline void getrf_batched(handle_t* h, int n, T** d_Aarray, int lda,
                          gt::blas::index_t* d_PivotArray, int* d_infoArray,
                          int batchSize);

#define CREATE_GETRF_BATCHED(METHOD, GTTYPE, BLASTYPE)                         \
  template <>                                                                  \
  inline void getrf_batched<GTTYPE>(handle_t * h, int n, GTTYPE** d_Aarray,    \
                                    int lda, gt::blas::index_t* d_PivotArray,  \
                                    int* d_infoArray, int batchSize)           \
  {                                                                            \
    gtGpuCheck((hipError_t)METHOD(h->handle, n, n,                             \
                                  reinterpret_cast<BLASTYPE**>(d_Aarray), lda, \
                                  d_PivotArray, n, d_infoArray, batchSize));   \
  }

CREATE_GETRF_BATCHED(rocsolver_zgetrf_batched, gt::complex<double>,
                     rocblas_double_complex)
CREATE_GETRF_BATCHED(rocsolver_cgetrf_batched, gt::complex<float>,
                     rocblas_float_complex)
CREATE_GETRF_BATCHED(rocsolver_dgetrf_batched, double, double)
CREATE_GETRF_BATCHED(rocsolver_sgetrf_batched, float, float)

#undef CREATE_GETRF_BATCHED

template <typename T>
inline void getrs_batched(handle_t* h, int n, int nrhs, T* const* d_Aarray,
                          int lda, gt::blas::index_t* devIpiv, T** d_Barray,
                          int ldb, int batchSize);

#define CREATE_GETRS_BATCHED(METHOD, GTTYPE, BLASTYPE)                         \
  template <>                                                                  \
  inline void getrs_batched<GTTYPE>(                                           \
    handle_t * h, int n, int nrhs, GTTYPE* const* d_Aarray, int lda,           \
    gt::blas::index_t* devIpiv, GTTYPE** d_Barray, int ldb, int batchSize)     \
  {                                                                            \
    gtGpuCheck((hipError_t)METHOD(                                             \
      h->handle, rocblas_operation_none, n, nrhs,                              \
      reinterpret_cast<BLASTYPE* const*>(d_Aarray), lda, devIpiv, n,           \
      reinterpret_cast<BLASTYPE**>(d_Barray), ldb, batchSize));                \
  }

CREATE_GETRS_BATCHED(rocsolver_zgetrs_batched, gt::complex<double>,
                     rocblas_double_complex)
CREATE_GETRS_BATCHED(rocsolver_cgetrs_batched, gt::complex<float>,
                     rocblas_float_complex)
CREATE_GETRS_BATCHED(rocsolver_dgetrs_batched, double, double)
CREATE_GETRS_BATCHED(rocsolver_sgetrs_batched, float, float)

#undef CREATE_GETRS_BATCHED

} // namespace blas

} // namespace gt

#endif
