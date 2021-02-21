#include "gtensor/blas.h"
#include "gtensor/cblas.h"

void gtblas_create(gt::blas::handle_t* h)
{
  gt::blas::create(h);
}

void gtblas_destroy(gt::blas::handle_t h)
{
  gt::blas::destroy(h);
}

void gtblas_set_stream(gt::blas::handle_t h, gt::blas::stream_t stream_id)
{
  gt::blas::set_stream(h, stream_id);
}

void gtblas_get_stream(gt::blas::handle_t h, gt::blas::stream_t* stream_id)
{
  gt::blas::get_stream(h, stream_id);
}

// ======================================================================
// gtblas_Xaxpy

#define CREATE_C_AXPY(CNAME, CPPTYPE)                                          \
  void CNAME(gt::blas::handle_t h, int n, CPPTYPE a, const CPPTYPE* x,         \
             int incx, CPPTYPE* y, int incy)                                   \
  {                                                                            \
    gt::blas::axpy(h, n, a, x, incx, y, incy);                                 \
  }

CREATE_C_AXPY(gtblas_saxpy, float)
CREATE_C_AXPY(gtblas_daxpy, double)
CREATE_C_AXPY(gtblas_caxpy, gt::complex<float>)
CREATE_C_AXPY(gtblas_zaxpy, gt::complex<double>)

#undef CREATE_C_AXPY

// ======================================================================
// gtblas_Xscal

#define CREATE_C_SCAL(CNAME, CPPTYPE)                                          \
  void CNAME(gt::blas::handle_t h, int n, CPPTYPE fac, CPPTYPE* arr, int incx) \
  {                                                                            \
    gt::blas::scal(h, n, fac, arr, incx);                                      \
  }

CREATE_C_SCAL(gtblas_sscal, float)
CREATE_C_SCAL(gtblas_dscal, double)
CREATE_C_SCAL(gtblas_cscal, gt::complex<float>)
CREATE_C_SCAL(gtblas_zscal, gt::complex<double>)

#undef CREATE_C_SCAL

// ======================================================================
// gtblas_Xcopy

#define CREATE_C_COPY(CNAME, CPPTYPE)                                          \
  void CNAME(gt::blas::handle_t h, int n, const CPPTYPE* x, int incx,          \
             CPPTYPE* y, int incy)                                             \
  {                                                                            \
    gt::blas::copy(h, n, x, incx, y, incy);                                    \
  }

CREATE_C_COPY(gtblas_scopy, float)
CREATE_C_COPY(gtblas_dcopy, double)
CREATE_C_COPY(gtblas_ccopy, gt::complex<float>)
CREATE_C_COPY(gtblas_zcopy, gt::complex<double>)

// ======================================================================
// gtblas_Xdot

#define CREATE_C_DOT(CNAME, CPPTYPE)                                           \
  void CNAME(gt::blas::handle_t h, int n, const CPPTYPE* x, int incx,          \
             CPPTYPE* y, int incy)                                             \
  {                                                                            \
    gt::blas::dot(h, n, x, incx, y, incy);                                     \
  }

CREATE_C_DOT(gtblas_sdot, float)
CREATE_C_DOT(gtblas_ddot, double)

#undef CREATE_C_DOT

// ======================================================================
// gtblas_Xdotu

#define CREATE_C_DOTU(CNAME, CPPTYPE)                                          \
  void CNAME(gt::blas::handle_t h, int n, const CPPTYPE* x, int incx,          \
             CPPTYPE* y, int incy)                                             \
  {                                                                            \
    gt::blas::dotu(h, n, x, incx, y, incy);                                    \
  }

CREATE_C_DOTU(gtblas_cdotu, gt::complex<float>)
CREATE_C_DOTU(gtblas_zdotu, gt::complex<double>)

#undef CREATE_C_DOTU

// ======================================================================
// gtblas_Xdotc

#define CREATE_C_DOTC(CNAME, CPPTYPE)                                          \
  void CNAME(gt::blas::handle_t h, int n, const CPPTYPE* x, int incx,          \
             CPPTYPE* y, int incy)                                             \
  {                                                                            \
    gt::blas::dotu(h, n, x, incx, y, incy);                                    \
  }

CREATE_C_DOTC(gtblas_cdotc, gt::complex<float>)
CREATE_C_DOTC(gtblas_zdotc, gt::complex<double>)

#undef CREATE_C_DOTC

// ======================================================================
// gtblas_Xgemv

#define CREATE_C_GEMV(CNAME, CPPTYPE)                                          \
  void CNAME(gt::blas::handle_t h, int m, int n, CPPTYPE alpha,                \
             const CPPTYPE* A, int lda, const CPPTYPE* x, int incx,            \
             CPPTYPE beta, CPPTYPE* y, int incy)                               \
  {                                                                            \
    gt::blas::gemv(h, m, n, alpha, A, lda, x, incx, beta, y, incy);            \
  }

CREATE_C_GEMV(gtblas_sgemv, float)
CREATE_C_GEMV(gtblas_dgemv, double)
CREATE_C_GEMV(gtblas_cgemv, gt::complex<float>)
CREATE_C_GEMV(gtblas_zgemv, gt::complex<double>)

#undef CREATE_C_GEMV

// ======================================================================
// gtblas_Xgetrf_batched

#define CREATE_C_GETRF_BATCHED(CNAME, CPPTYPE)                                 \
  void CNAME(gt::blas::handle_t h, int n, CPPTYPE** d_Aarray, int lda,         \
             gt::blas::index_t* d_PivotArray, int* d_infoArray, int batchSize) \
  {                                                                            \
    gt::blas::getrf_batched(h, n, d_Aarray, lda, d_PivotArray, d_infoArray,    \
                            batchSize);                                        \
  }

CREATE_C_GETRF_BATCHED(gtblas_sgetrf_batched, float)
CREATE_C_GETRF_BATCHED(gtblas_dgetrf_batched, double)
CREATE_C_GETRF_BATCHED(gtblas_cgetrf_batched, gt::complex<float>)
CREATE_C_GETRF_BATCHED(gtblas_zgetrf_batched, gt::complex<double>)

#undef CREATE_C_GETRF_BATCHED

// ======================================================================
// gtblas_Xgetrs_batched

#define CREATE_C_GETRS_BATCHED(CNAME, CPPTYPE)                                 \
  void CNAME(gt::blas::handle_t h, int n, int nrhs, CPPTYPE** d_Aarray,        \
             int lda, gt::blas::index_t* d_PivotArray, CPPTYPE** d_Barray,     \
             int ldb, int batchSize)                                           \
  {                                                                            \
    gt::blas::getrs_batched(h, n, nrhs, d_Aarray, lda, d_PivotArray, d_Barray, \
                            ldb, batchSize);                                   \
  }

CREATE_C_GETRS_BATCHED(gtblas_sgetrs_batched, float)
CREATE_C_GETRS_BATCHED(gtblas_dgetrs_batched, double)
CREATE_C_GETRS_BATCHED(gtblas_cgetrs_batched, gt::complex<float>)
CREATE_C_GETRS_BATCHED(gtblas_zgetrs_batched, gt::complex<double>)

#undef CREATE_C_GETRS_BATCHED
