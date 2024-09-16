#ifndef GTENSOR_BLAS_HIP_H
#define GTENSOR_BLAS_HIP_H

#include "rocblas/rocblas.h"
#include "rocsolver/rocsolver.h"

// ======================================================================
// error handling helper

#define gtBlasCheck(what)                                                      \
  {                                                                            \
    gtBlasCheckImpl(what, __FILE__, __LINE__);                                 \
  }

inline void gtBlasCheckImpl(rocblas_status code, const char* file, int line)
{
  if (code != rocblas_status_success) {
    fprintf(stderr, "gtBlasCheck: rocblas status %d at %s:%d\n", code, file,
            line);
    abort();
  }
}

namespace gt
{

namespace blas
{

// ======================================================================
// types aliases

using index_t = int;

// ======================================================================
// handle and stream management

class handle_hip : public detail::handle_base<handle_hip, rocblas_handle>
{
public:
  handle_hip() { gtBlasCheck(rocblas_create_handle(&handle_)); }
  ~handle_hip() { gtBlasCheck(rocblas_destroy_handle(handle_)); }

  void set_stream(gt::stream_view sview)
  {
    gtBlasCheck(rocblas_set_stream(handle_, sview.get_backend_stream()));
  }

  gt::stream_view get_stream()
  {
    hipStream_t s;
    gtBlasCheck(rocblas_get_stream(handle_, &s));
    return gt::stream_view{s};
  }
};

using handle_t = handle_hip;

// ======================================================================
// axpy

template <typename T>
inline void axpy(handle_t& h, int n, T a, const T* x, int incx, T* y, int incy);

#define CREATE_AXPY(METHOD, GTTYPE, BLASTYPE)                                  \
  template <>                                                                  \
  inline void axpy<GTTYPE>(handle_t & h, int n, GTTYPE a, const GTTYPE* x,     \
                           int incx, GTTYPE* y, int incy)                      \
  {                                                                            \
    gtBlasCheck(METHOD(h.get_backend_handle(), n,                              \
                       reinterpret_cast<BLASTYPE*>(&a),                        \
                       reinterpret_cast<const BLASTYPE*>(x), incx,             \
                       reinterpret_cast<BLASTYPE*>(y), incy));                 \
  }

CREATE_AXPY(rocblas_zaxpy, gt::complex<double>, rocblas_double_complex)
CREATE_AXPY(rocblas_caxpy, gt::complex<float>, rocblas_float_complex)
CREATE_AXPY(rocblas_daxpy, double, double)
CREATE_AXPY(rocblas_saxpy, float, float)

#undef CREATE_AXPY

// ======================================================================
// scal

template <typename S, typename T>
inline void scal(handle_t& h, int n, S fac, T* arr, const int incx);

#define CREATE_SCAL(METHOD, GTTYPE, BLASTYPE)                                  \
  template <>                                                                  \
  inline void scal<GTTYPE, GTTYPE>(handle_t & h, int n, GTTYPE fac,            \
                                   GTTYPE* arr, const int incx)                \
  {                                                                            \
    gtBlasCheck(METHOD(h.get_backend_handle(), n,                              \
                       reinterpret_cast<BLASTYPE*>(&fac),                      \
                       reinterpret_cast<BLASTYPE*>(arr), incx));               \
  }

CREATE_SCAL(rocblas_zscal, gt::complex<double>, rocblas_double_complex)
CREATE_SCAL(rocblas_cscal, gt::complex<float>, rocblas_float_complex)
CREATE_SCAL(rocblas_dscal, double, double)
CREATE_SCAL(rocblas_sscal, float, float)

#undef CREATE_SCAL

// ======================================================================
// (zd|cs)scal

template <>
inline void scal<double, gt::complex<double>>(handle_t& h, int n, double fac,
                                              gt::complex<double>* arr,
                                              const int incx)
{
  gtBlasCheck(rocblas_zdscal(h.get_backend_handle(), n, &fac,
                             reinterpret_cast<rocblas_double_complex*>(arr),
                             incx));
}

template <>
inline void scal<float, gt::complex<float>>(handle_t& h, int n, float fac,
                                            gt::complex<float>* arr,
                                            const int incx)
{
  gtBlasCheck(rocblas_csscal(h.get_backend_handle(), n, &fac,
                             reinterpret_cast<rocblas_float_complex*>(arr),
                             incx));
}

// ======================================================================
// copy

template <typename T>
inline void copy(handle_t& h, int n, const T* x, int incx, T* y, int incy);

#define CREATE_COPY(METHOD, GTTYPE, BLASTYPE)                                  \
  template <>                                                                  \
  inline void copy<GTTYPE>(handle_t & h, int n, const GTTYPE* x, int incx,     \
                           GTTYPE* y, int incy)                                \
  {                                                                            \
    gtBlasCheck(METHOD(h.get_backend_handle(), n,                              \
                       reinterpret_cast<const BLASTYPE*>(x), incx,             \
                       reinterpret_cast<BLASTYPE*>(y), incy));                 \
  }

CREATE_COPY(rocblas_zcopy, gt::complex<double>, rocblas_double_complex)
CREATE_COPY(rocblas_ccopy, gt::complex<float>, rocblas_float_complex)
CREATE_COPY(rocblas_dcopy, double, double)
CREATE_COPY(rocblas_scopy, float, float)

#undef CREATE_COPY

// ======================================================================
// dot, dotc (conjugate)

template <typename T>
inline T dot(handle_t& h, int n, const T* x, int incx, const T* y, int incy);

#define CREATE_DOT(METHOD, GTTYPE, BLASTYPE)                                   \
  template <>                                                                  \
  inline GTTYPE dot<GTTYPE>(handle_t & h, int n, const GTTYPE* x, int incx,    \
                            const GTTYPE* y, int incy)                         \
  {                                                                            \
    GTTYPE result;                                                             \
    gtBlasCheck(METHOD(h.get_backend_handle(), n,                              \
                       reinterpret_cast<const BLASTYPE*>(x), incx,             \
                       reinterpret_cast<const BLASTYPE*>(y), incy,             \
                       reinterpret_cast<BLASTYPE*>(&result)));                 \
    return result;                                                             \
  }

CREATE_DOT(rocblas_ddot, double, double)
CREATE_DOT(rocblas_sdot, float, float)

#undef CREATE_DOT

template <typename T>
inline T dotu(handle_t& h, int n, const T* x, int incx, const T* y, int incy);

#define CREATE_DOTU(METHOD, GTTYPE, BLASTYPE)                                  \
  template <>                                                                  \
  inline GTTYPE dotu<GTTYPE>(handle_t & h, int n, const GTTYPE* x, int incx,   \
                             const GTTYPE* y, int incy)                        \
  {                                                                            \
    GTTYPE result;                                                             \
    gtBlasCheck(METHOD(h.get_backend_handle(), n,                              \
                       reinterpret_cast<const BLASTYPE*>(x), incx,             \
                       reinterpret_cast<const BLASTYPE*>(y), incy,             \
                       reinterpret_cast<BLASTYPE*>(&result)));                 \
    return result;                                                             \
  }

CREATE_DOTU(rocblas_zdotu, gt::complex<double>, rocblas_double_complex)
CREATE_DOTU(rocblas_cdotu, gt::complex<float>, rocblas_float_complex)

#undef CREATE_DOTU

template <typename T>
inline T dotc(handle_t& h, int n, const T* x, int incx, const T* y, int incy);

#define CREATE_DOTC(METHOD, GTTYPE, BLASTYPE)                                  \
  template <>                                                                  \
  inline GTTYPE dotc<GTTYPE>(handle_t & h, int n, const GTTYPE* x, int incx,   \
                             const GTTYPE* y, int incy)                        \
  {                                                                            \
    GTTYPE result;                                                             \
    gtBlasCheck(METHOD(h.get_backend_handle(), n,                              \
                       reinterpret_cast<const BLASTYPE*>(x), incx,             \
                       reinterpret_cast<const BLASTYPE*>(y), incy,             \
                       reinterpret_cast<BLASTYPE*>(&result)));                 \
    return result;                                                             \
  }

CREATE_DOTC(rocblas_zdotc, gt::complex<double>, rocblas_double_complex)
CREATE_DOTC(rocblas_cdotc, gt::complex<float>, rocblas_float_complex)

#undef CREATE_DOTC

// ======================================================================
// gemv

template <typename T>
inline void gemv(handle_t& h, int m, int n, T alpha, const T* A, int lda,
                 const T* x, int incx, T beta, T* y, int incy);

#define CREATE_GEMV(METHOD, GTTYPE, BLASTYPE)                                  \
  template <>                                                                  \
  inline void gemv<GTTYPE>(handle_t & h, int m, int n, GTTYPE alpha,           \
                           const GTTYPE* A, int lda, const GTTYPE* x,          \
                           int incx, GTTYPE beta, GTTYPE* y, int incy)         \
  {                                                                            \
    gtBlasCheck(METHOD(h.get_backend_handle(), rocblas_operation_none, m, n,   \
                       reinterpret_cast<BLASTYPE*>(&alpha),                    \
                       reinterpret_cast<const BLASTYPE*>(A), lda,              \
                       reinterpret_cast<const BLASTYPE*>(x), incx,             \
                       reinterpret_cast<BLASTYPE*>(&beta),                     \
                       reinterpret_cast<BLASTYPE*>(y), incy));                 \
  }

CREATE_GEMV(rocblas_zgemv, gt::complex<double>, rocblas_double_complex)
CREATE_GEMV(rocblas_cgemv, gt::complex<float>, rocblas_float_complex)
CREATE_GEMV(rocblas_dgemv, double, double)
CREATE_GEMV(rocblas_sgemv, float, float)

#undef CREATE_GEMV

// ======================================================================
// getrf/getrs batched

template <typename T>
inline void getrf_batched(handle_t& h, int n, T** d_Aarray, int lda,
                          gt::blas::index_t* d_PivotArray, int* d_infoArray,
                          int batchSize);

#define CREATE_GETRF_BATCHED(METHOD, GTTYPE, BLASTYPE)                         \
  template <>                                                                  \
  inline void getrf_batched<GTTYPE>(handle_t & h, int n, GTTYPE** d_Aarray,    \
                                    int lda, gt::blas::index_t* d_PivotArray,  \
                                    int* d_infoArray, int batchSize)           \
  {                                                                            \
    gtBlasCheck(METHOD(h.get_backend_handle(), n, n,                           \
                       reinterpret_cast<BLASTYPE**>(d_Aarray), lda,            \
                       d_PivotArray, n, d_infoArray, batchSize));              \
  }

CREATE_GETRF_BATCHED(rocsolver_zgetrf_batched, gt::complex<double>,
                     rocblas_double_complex)
CREATE_GETRF_BATCHED(rocsolver_cgetrf_batched, gt::complex<float>,
                     rocblas_float_complex)
CREATE_GETRF_BATCHED(rocsolver_dgetrf_batched, double, double)
CREATE_GETRF_BATCHED(rocsolver_sgetrf_batched, float, float)

#undef CREATE_GETRF_BATCHED

template <typename T>
inline void getrs_batched(handle_t& h, int n, int nrhs, T* const* d_Aarray,
                          int lda, gt::blas::index_t* devIpiv, T** d_Barray,
                          int ldb, int batchSize);

#define CREATE_GETRS_BATCHED(METHOD, GTTYPE, BLASTYPE)                         \
  template <>                                                                  \
  inline void getrs_batched<GTTYPE>(                                           \
    handle_t & h, int n, int nrhs, GTTYPE* const* d_Aarray, int lda,           \
    gt::blas::index_t* devIpiv, GTTYPE** d_Barray, int ldb, int batchSize)     \
  {                                                                            \
    gtBlasCheck(                                                               \
      METHOD(h.get_backend_handle(), rocblas_operation_none, n, nrhs,          \
             reinterpret_cast<BLASTYPE* const*>(d_Aarray), lda, devIpiv, n,    \
             reinterpret_cast<BLASTYPE**>(d_Barray), ldb, batchSize));         \
  }

CREATE_GETRS_BATCHED(rocsolver_zgetrs_batched, gt::complex<double>,
                     rocblas_double_complex)
CREATE_GETRS_BATCHED(rocsolver_cgetrs_batched, gt::complex<float>,
                     rocblas_float_complex)
CREATE_GETRS_BATCHED(rocsolver_dgetrs_batched, double, double)
CREATE_GETRS_BATCHED(rocsolver_sgetrs_batched, float, float)

#undef CREATE_GETRS_BATCHED

// ======================================================================
// getrf/getrs batched without pivoting

template <typename T>
inline void getrf_npvt_batched(handle_t& h, int n, T** d_Aarray, int lda,
                               int* d_infoArray, int batchSize);

#define CREATE_GETRF_NPVT_BATCHED(METHOD, GTTYPE, BLASTYPE)                    \
  template <>                                                                  \
  inline void getrf_npvt_batched<GTTYPE>(handle_t & h, int n,                  \
                                         GTTYPE** d_Aarray, int lda,           \
                                         int* d_infoArray, int batchSize)      \
  {                                                                            \
    gtBlasCheck(METHOD(h.get_backend_handle(), n, n,                           \
                       reinterpret_cast<BLASTYPE**>(d_Aarray), lda,            \
                       d_infoArray, batchSize));                               \
  }

CREATE_GETRF_NPVT_BATCHED(rocsolver_zgetrf_npvt_batched, gt::complex<double>,
                          rocblas_double_complex)
CREATE_GETRF_NPVT_BATCHED(rocsolver_cgetrf_npvt_batched, gt::complex<float>,
                          rocblas_float_complex)
CREATE_GETRF_NPVT_BATCHED(rocsolver_dgetrf_npvt_batched, double, double)
CREATE_GETRF_NPVT_BATCHED(rocsolver_sgetrf_npvt_batched, float, float)

#undef CREATE_GETRF_NPVT_BATCHED

// ======================================================================
// getri batched

template <typename T>
inline void getri_batched(handle_t& h, int n, T* const* d_Aarray, int lda,
                          gt::blas::index_t* devIpiv, T** d_Carray, int ldc,
                          int* d_infoArray, int batchSize);

#define CREATE_GETRI_BATCHED(METHOD, GTTYPE, BLASTYPE)                         \
  template <>                                                                  \
  inline void getri_batched<GTTYPE>(                                           \
    handle_t & h, int n, GTTYPE* const* d_Aarray, int lda,                     \
    gt::blas::index_t* devIpiv, GTTYPE** d_Carray, int ldc, int* d_infoArray,  \
    int batchSize)                                                             \
  {                                                                            \
    gtBlasCheck(METHOD(                                                        \
      h.get_backend_handle(), n, reinterpret_cast<BLASTYPE* const*>(d_Aarray), \
      lda, devIpiv, n, reinterpret_cast<BLASTYPE**>(d_Carray), ldc,            \
      reinterpret_cast<rocblas_int*>(d_infoArray), batchSize));                \
  }

CREATE_GETRI_BATCHED(rocsolver_zgetri_outofplace_batched, gt::complex<double>,
                     rocblas_double_complex)
CREATE_GETRI_BATCHED(rocsolver_cgetri_outofplace_batched, gt::complex<float>,
                     rocblas_float_complex)
CREATE_GETRI_BATCHED(rocsolver_dgetri_outofplace_batched, double, double)
CREATE_GETRI_BATCHED(rocsolver_sgetri_outofplace_batched, float, float)

#undef CREATE_GETRI_BATCHED

// ======================================================================
// gemm batched

template <typename T>
inline void gemm_batched(handle_t& h, int m, int n, int k, T alpha,
                         T** d_Aarray, int lda, T** d_Barray, int ldb, T beta,
                         T** d_Carray, int ldc, int batchSize);

#define CREATE_GEMM_BATCHED(METHOD, GTTYPE, BLASTYPE)                          \
  template <>                                                                  \
  inline void gemm_batched<GTTYPE>(handle_t & h, int m, int n, int k,          \
                                   GTTYPE alpha, GTTYPE** d_Aarray, int lda,   \
                                   GTTYPE** d_Barray, int ldb, GTTYPE beta,    \
                                   GTTYPE** d_Carray, int ldc, int batchSize)  \
  {                                                                            \
    gtBlasCheck(METHOD(                                                        \
      h.get_backend_handle(), rocblas_operation_none, rocblas_operation_none,  \
      m, n, k, reinterpret_cast<BLASTYPE*>(&alpha),                            \
      reinterpret_cast<BLASTYPE**>(d_Aarray), lda,                             \
      reinterpret_cast<BLASTYPE**>(d_Barray), ldb,                             \
      reinterpret_cast<BLASTYPE*>(&beta),                                      \
      reinterpret_cast<BLASTYPE**>(d_Carray), ldc, batchSize));                \
  }

CREATE_GEMM_BATCHED(rocblas_zgemm_batched, gt::complex<double>,
                    rocblas_double_complex)
CREATE_GEMM_BATCHED(rocblas_cgemm_batched, gt::complex<float>,
                    rocblas_float_complex)
CREATE_GEMM_BATCHED(rocblas_dgemm_batched, double, double);
CREATE_GEMM_BATCHED(rocblas_sgemm_batched, float, float);

#undef CREATE_GEMM_BATCHED

} // namespace blas

} // namespace gt

#endif
