#include "gtensor/blas.h"
#include "gtensor/cblas.h"

static gt::blas::handle_t* g_handle = nullptr;

void gtblas_create()
{
  if (g_handle == nullptr) {
    g_handle = gt::blas::create();
  }
}

void gtblas_destroy()
{
  if (g_handle != nullptr) {
    gt::blas::destroy(g_handle);
    g_handle = nullptr;
  }
}

void gtblas_set_stream(gt::blas::stream_t stream_id)
{
  gt::blas::set_stream(g_handle, stream_id);
}

void gtblas_get_stream(gt::blas::stream_t* stream_id)
{
  gt::blas::get_stream(g_handle, stream_id);
}

// ======================================================================
// gtblas_Xaxpy

#define CREATE_C_AXPY(CNAME, CPPTYPE)                                          \
  void CNAME(int n, const CPPTYPE* a, const CPPTYPE* x, int incx, CPPTYPE* y,  \
             int incy)                                                         \
  {                                                                            \
    gt::blas::axpy(g_handle, n, *a, x, incx, y, incy);                         \
  }

CREATE_C_AXPY(gtblas_saxpy, float)
CREATE_C_AXPY(gtblas_daxpy, double)
CREATE_C_AXPY(gtblas_caxpy, gt::complex<float>)
CREATE_C_AXPY(gtblas_zaxpy, gt::complex<double>)

#undef CREATE_C_AXPY

// ======================================================================
// gtblas_Xscal

#define CREATE_C_SCAL(CNAME, STYPE, ATYPE)                                     \
  void CNAME(int n, const STYPE* a, ATYPE* x, int incx)                        \
  {                                                                            \
    gt::blas::scal(g_handle, n, *a, x, incx);                                  \
  }

CREATE_C_SCAL(gtblas_sscal, float, float)
CREATE_C_SCAL(gtblas_dscal, double, double)
CREATE_C_SCAL(gtblas_cscal, gt::complex<float>, gt::complex<float>)
CREATE_C_SCAL(gtblas_zscal, gt::complex<double>, gt::complex<double>)
CREATE_C_SCAL(gtblas_csscal, float, gt::complex<float>)
CREATE_C_SCAL(gtblas_zdscal, double, gt::complex<double>)

#undef CREATE_C_SCAL

// ======================================================================
// gtblas_Xcopy

#define CREATE_C_COPY(CNAME, CPPTYPE)                                          \
  void CNAME(int n, const CPPTYPE* x, int incx, CPPTYPE* y, int incy)          \
  {                                                                            \
    gt::blas::copy(g_handle, n, x, incx, y, incy);                             \
  }

CREATE_C_COPY(gtblas_scopy, float)
CREATE_C_COPY(gtblas_dcopy, double)
CREATE_C_COPY(gtblas_ccopy, gt::complex<float>)
CREATE_C_COPY(gtblas_zcopy, gt::complex<double>)

// ======================================================================
// gtblas_Xdot

#define CREATE_C_DOT(CNAME, CPPTYPE)                                           \
  void CNAME(int n, const CPPTYPE* x, int incx, CPPTYPE* y, int incy)          \
  {                                                                            \
    gt::blas::dot(g_handle, n, x, incx, y, incy);                              \
  }

CREATE_C_DOT(gtblas_sdot, float)
CREATE_C_DOT(gtblas_ddot, double)

#undef CREATE_C_DOT

// ======================================================================
// gtblas_Xdotu

#define CREATE_C_DOTU(CNAME, CPPTYPE)                                          \
  void CNAME(int n, const CPPTYPE* x, int incx, CPPTYPE* y, int incy)          \
  {                                                                            \
    gt::blas::dotu(g_handle, n, x, incx, y, incy);                             \
  }

CREATE_C_DOTU(gtblas_cdotu, gt::complex<float>)
CREATE_C_DOTU(gtblas_zdotu, gt::complex<double>)

#undef CREATE_C_DOTU

// ======================================================================
// gtblas_Xdotc

#define CREATE_C_DOTC(CNAME, CPPTYPE)                                          \
  void CNAME(int n, const CPPTYPE* x, int incx, CPPTYPE* y, int incy)          \
  {                                                                            \
    gt::blas::dotu(g_handle, n, x, incx, y, incy);                             \
  }

CREATE_C_DOTC(gtblas_cdotc, gt::complex<float>)
CREATE_C_DOTC(gtblas_zdotc, gt::complex<double>)

#undef CREATE_C_DOTC

// ======================================================================
// gtblas_Xgemv

#define CREATE_C_GEMV(CNAME, CPPTYPE)                                          \
  void CNAME(int m, int n, const CPPTYPE* alpha, const CPPTYPE* A, int lda,    \
             const CPPTYPE* x, int incx, const CPPTYPE* beta, CPPTYPE* y,      \
             int incy)                                                         \
  {                                                                            \
    gt::blas::gemv(g_handle, m, n, *alpha, A, lda, x, incx, *beta, y, incy);   \
  }

CREATE_C_GEMV(gtblas_sgemv, float)
CREATE_C_GEMV(gtblas_dgemv, double)
CREATE_C_GEMV(gtblas_cgemv, gt::complex<float>)
CREATE_C_GEMV(gtblas_zgemv, gt::complex<double>)

#undef CREATE_C_GEMV

// ======================================================================
// gtblas_Xgetrf_batched

#define CREATE_C_GETRF_BATCHED(CNAME, CPPTYPE)                                 \
  void CNAME(int n, CPPTYPE** d_Aarray, int lda,                               \
             gt::blas::index_t* d_PivotArray, int* d_infoArray, int batchSize) \
  {                                                                            \
    gt::blas::getrf_batched(g_handle, n, d_Aarray, lda, d_PivotArray,          \
                            d_infoArray, batchSize);                           \
  }

CREATE_C_GETRF_BATCHED(gtblas_sgetrf_batched, float)
CREATE_C_GETRF_BATCHED(gtblas_dgetrf_batched, double)
CREATE_C_GETRF_BATCHED(gtblas_cgetrf_batched, gt::complex<float>)
CREATE_C_GETRF_BATCHED(gtblas_zgetrf_batched, gt::complex<double>)

#undef CREATE_C_GETRF_BATCHED

// ======================================================================
// gtblas_Xgetrs_batched

#define CREATE_C_GETRS_BATCHED(CNAME, CPPTYPE)                                 \
  void CNAME(int n, int nrhs, CPPTYPE** d_Aarray, int lda,                     \
             gt::blas::index_t* d_PivotArray, CPPTYPE** d_Barray, int ldb,     \
             int batchSize)                                                    \
  {                                                                            \
    gt::blas::getrs_batched(g_handle, n, nrhs, d_Aarray, lda, d_PivotArray,    \
                            d_Barray, ldb, batchSize);                         \
  }

CREATE_C_GETRS_BATCHED(gtblas_sgetrs_batched, float)
CREATE_C_GETRS_BATCHED(gtblas_dgetrs_batched, double)
CREATE_C_GETRS_BATCHED(gtblas_cgetrs_batched, gt::complex<float>)
CREATE_C_GETRS_BATCHED(gtblas_zgetrs_batched, gt::complex<double>)

#undef CREATE_C_GETRS_BATCHED
