#include <cassert>
#include <cstdint>

#include "gt-blas/blas.h"
#include "gt-blas/cblas.h"

static gt::blas::handle_t* g_handle = nullptr;

namespace detail
{

template <typename T>
inline void assert_aligned(T* p, size_t alignment)
{
  assert((uintptr_t)(void*)p % alignment == 0);
}

template <typename T>
inline T fc2cpp_deref(const T* f)
{
  return *f;
}

// NB: convert to the gt type component-wise in case the alignment
// requirements are different, which will be the case when the thrust
// storage backend is used.
template <typename T>
inline gt::complex<T> fc2cpp_deref(const f2c_complex<T>* c)
{
  return gt::complex<T>(c->real(), c->imag());
}

template <typename T>
inline T* cast_aligned(T* f)
{
  assert_aligned(f, sizeof(T));
  return f;
}

template <typename T>
inline T** cast_aligned(T** f)
{
  assert_aligned(f[0], sizeof(T));
  return f;
}

template <typename T>
inline auto cast_aligned(f2c_complex<T>* c)
{
  assert_aligned(c, 2 * sizeof(T));
  return reinterpret_cast<gt::complex<T>*>(c);
}

template <typename T>
inline auto cast_aligned(f2c_complex<T>** c)
{
  assert_aligned(c[0], 2 * sizeof(T));
  return reinterpret_cast<gt::complex<T>**>(c);
}

template <typename T>
inline auto cast_aligned(const f2c_complex<T>* c)
{
  assert_aligned(c, 2 * sizeof(T));
  return reinterpret_cast<const gt::complex<T>*>(c);
}

template <typename T>
inline auto cast_aligned(const f2c_complex<T>** c)
{
  assert_aligned(c[0], 2 * sizeof(T));
  return reinterpret_cast<const gt::complex<T>**>(c);
}

} // namespace detail

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
    gt::blas::axpy(g_handle, n, detail::fc2cpp_deref(a),                       \
                   detail::cast_aligned(x), incx, detail::cast_aligned(y),     \
                   incy);                                                      \
  }

CREATE_C_AXPY(gtblas_saxpy, float)
CREATE_C_AXPY(gtblas_daxpy, double)
CREATE_C_AXPY(gtblas_caxpy, f2c_complex<float>)
CREATE_C_AXPY(gtblas_zaxpy, f2c_complex<double>)

#undef CREATE_C_AXPY

// ======================================================================
// gtblas_Xscal

#define CREATE_C_SCAL(CNAME, STYPE, ATYPE)                                     \
  void CNAME(int n, const STYPE* a, ATYPE* x, int incx)                        \
  {                                                                            \
    gt::blas::scal(g_handle, n, detail::fc2cpp_deref(a),                       \
                   detail::cast_aligned(x), incx);                             \
  }

CREATE_C_SCAL(gtblas_sscal, float, float)
CREATE_C_SCAL(gtblas_dscal, double, double)
CREATE_C_SCAL(gtblas_cscal, f2c_complex<float>, f2c_complex<float>)
CREATE_C_SCAL(gtblas_zscal, f2c_complex<double>, f2c_complex<double>)
CREATE_C_SCAL(gtblas_csscal, float, f2c_complex<float>)
CREATE_C_SCAL(gtblas_zdscal, double, f2c_complex<double>)

#undef CREATE_C_SCAL

// ======================================================================
// gtblas_Xcopy

#define CREATE_C_COPY(CNAME, CPPTYPE)                                          \
  void CNAME(int n, const CPPTYPE* x, int incx, CPPTYPE* y, int incy)          \
  {                                                                            \
    gt::blas::copy(g_handle, n, detail::cast_aligned(x), incx,                 \
                   detail::cast_aligned(y), incy);                             \
  }

CREATE_C_COPY(gtblas_scopy, float)
CREATE_C_COPY(gtblas_dcopy, double)
CREATE_C_COPY(gtblas_ccopy, f2c_complex<float>)
CREATE_C_COPY(gtblas_zcopy, f2c_complex<double>)

// ======================================================================
// gtblas_Xdot

#define CREATE_C_DOT(CNAME, CPPTYPE)                                           \
  void CNAME(int n, const CPPTYPE* x, int incx, CPPTYPE* y, int incy)          \
  {                                                                            \
    gt::blas::dot(g_handle, n, detail::cast_aligned(x), incx,                  \
                  detail::cast_aligned(y), incy);                              \
  }

CREATE_C_DOT(gtblas_sdot, float)
CREATE_C_DOT(gtblas_ddot, double)

#undef CREATE_C_DOT

// ======================================================================
// gtblas_Xdotu

#define CREATE_C_DOTU(CNAME, CPPTYPE)                                          \
  void CNAME(int n, const CPPTYPE* x, int incx, CPPTYPE* y, int incy)          \
  {                                                                            \
    gt::blas::dotu(g_handle, n, detail::cast_aligned(x), incx,                 \
                   detail::cast_aligned(y), incy);                             \
  }

CREATE_C_DOTU(gtblas_cdotu, f2c_complex<float>)
CREATE_C_DOTU(gtblas_zdotu, f2c_complex<double>)

#undef CREATE_C_DOTU

// ======================================================================
// gtblas_Xdotc

#define CREATE_C_DOTC(CNAME, CPPTYPE)                                          \
  void CNAME(int n, const CPPTYPE* x, int incx, CPPTYPE* y, int incy)          \
  {                                                                            \
    gt::blas::dotu(g_handle, n, detail::cast_aligned(x), incx,                 \
                   detail::cast_aligned(y), incy);                             \
  }

CREATE_C_DOTC(gtblas_cdotc, f2c_complex<float>)
CREATE_C_DOTC(gtblas_zdotc, f2c_complex<double>)

#undef CREATE_C_DOTC

// ======================================================================
// gtblas_Xgemv

#define CREATE_C_GEMV(CNAME, CPPTYPE)                                          \
  void CNAME(int m, int n, const CPPTYPE* alpha, const CPPTYPE* A, int lda,    \
             const CPPTYPE* x, int incx, const CPPTYPE* beta, CPPTYPE* y,      \
             int incy)                                                         \
  {                                                                            \
    gt::blas::gemv(g_handle, m, n, detail::fc2cpp_deref(alpha),                \
                   detail::cast_aligned(A), lda, detail::cast_aligned(x),      \
                   incx, detail::fc2cpp_deref(beta), detail::cast_aligned(y),  \
                   incy);                                                      \
  }

CREATE_C_GEMV(gtblas_sgemv, float)
CREATE_C_GEMV(gtblas_dgemv, double)
CREATE_C_GEMV(gtblas_cgemv, f2c_complex<float>)
CREATE_C_GEMV(gtblas_zgemv, f2c_complex<double>)

#undef CREATE_C_GEMV

// ======================================================================
// gtblas_Xgetrf_batched

#define CREATE_C_GETRF_BATCHED(CNAME, CPPTYPE)                                 \
  void CNAME(int n, CPPTYPE** d_Aarray, int lda,                               \
             gt::blas::index_t* d_PivotArray, int* d_infoArray, int batchSize) \
  {                                                                            \
    gt::blas::getrf_batched(g_handle, n, detail::cast_aligned(d_Aarray), lda,  \
                            d_PivotArray, d_infoArray, batchSize);             \
  }

CREATE_C_GETRF_BATCHED(gtblas_sgetrf_batched, float)
CREATE_C_GETRF_BATCHED(gtblas_dgetrf_batched, double)
CREATE_C_GETRF_BATCHED(gtblas_cgetrf_batched, f2c_complex<float>)
CREATE_C_GETRF_BATCHED(gtblas_zgetrf_batched, f2c_complex<double>)

#undef CREATE_C_GETRF_BATCHED

// ======================================================================
// gtblas_Xgetrs_batched

#define CREATE_C_GETRS_BATCHED(CNAME, CPPTYPE)                                 \
  void CNAME(int n, int nrhs, CPPTYPE** d_Aarray, int lda,                     \
             gt::blas::index_t* d_PivotArray, CPPTYPE** d_Barray, int ldb,     \
             int batchSize)                                                    \
  {                                                                            \
    gt::blas::getrs_batched(g_handle, n, nrhs, detail::cast_aligned(d_Aarray), \
                            lda, d_PivotArray, detail::cast_aligned(d_Barray), \
                            ldb, batchSize);                                   \
  }

CREATE_C_GETRS_BATCHED(gtblas_sgetrs_batched, float)
CREATE_C_GETRS_BATCHED(gtblas_dgetrs_batched, double)
CREATE_C_GETRS_BATCHED(gtblas_cgetrs_batched, f2c_complex<float>)
CREATE_C_GETRS_BATCHED(gtblas_zgetrs_batched, f2c_complex<double>)

#undef CREATE_C_GETRS_BATCHED

// ======================================================================
// gtblas_banded_Xgetrs_batched

#define CREATE_C_BANDED_GETRS_BATCHED(CNAME, CPPTYPE)                          \
  void CNAME(int n, int nrhs, CPPTYPE** d_Aarray, int lda,                     \
             gt::blas::index_t* d_PivotArray, CPPTYPE** d_Barray, int ldb,     \
             int batchSize, int lbw, int ubw)                                  \
  {                                                                            \
    gt::blas::getrs_banded_batched(                                            \
      n, nrhs, detail::cast_aligned(d_Aarray), lda, d_PivotArray,              \
      detail::cast_aligned(d_Barray), ldb, batchSize, lbw, ubw);               \
  }

CREATE_C_BANDED_GETRS_BATCHED(gtblas_banded_sgetrs_batched, float)
CREATE_C_BANDED_GETRS_BATCHED(gtblas_banded_dgetrs_batched, double)
CREATE_C_BANDED_GETRS_BATCHED(gtblas_banded_cgetrs_batched, f2c_complex<float>)
CREATE_C_BANDED_GETRS_BATCHED(gtblas_banded_zgetrs_batched, f2c_complex<double>)

#undef CREATE_C_BANDED_GETRS_BATCHED

// ======================================================================
// gtblas_Xget_max_bandwidth

#define CREATE_C_GET_MAX_BANDWIDTH(CNAME, CPPTYPE)                             \
  void CNAME(int n, CPPTYPE** d_Aarray, int lda, int batchSize, int* lbw,      \
             int* ubw)                                                         \
  {                                                                            \
    auto bw = gt::blas::get_max_bandwidth(n, detail::cast_aligned(d_Aarray),   \
                                          lda, batchSize);                     \
    *lbw = bw.lower;                                                           \
    *ubw = bw.upper;                                                           \
  }

CREATE_C_GET_MAX_BANDWIDTH(gtblas_sget_max_bandwidth, float)
CREATE_C_GET_MAX_BANDWIDTH(gtblas_dget_max_bandwidth, double)
CREATE_C_GET_MAX_BANDWIDTH(gtblas_cget_max_bandwidth, f2c_complex<float>)
CREATE_C_GET_MAX_BANDWIDTH(gtblas_zget_max_bandwidth, f2c_complex<double>)

// ======================================================================
// gtblas_Xgetrf_npvt_batched
#define CREATE_C_GETRF_NPVT_BATCHED(CNAME, CPPTYPE)                            \
  void CNAME(int n, CPPTYPE** d_Aarray, int lda, int* d_infoArray,             \
             int batchSize)                                                    \
  {                                                                            \
    gt::blas::getrf_npvt_batched(g_handle, n, detail::cast_aligned(d_Aarray),  \
                                 lda, d_infoArray, batchSize);                 \
  }

CREATE_C_GETRF_NPVT_BATCHED(gtblas_cgetrf_npvt_batched, f2c_complex<float>)
CREATE_C_GETRF_NPVT_BATCHED(gtblas_zgetrf_npvt_batched, f2c_complex<double>)

#undef CREATE_C_GETRF_NPVT_BATCHED
