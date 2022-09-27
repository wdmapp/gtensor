#ifndef GTENSOR_BLAS_HOST_H
#define GTENSOR_BLAS_HOST_H

#include <gtensor/gtensor.h>

#include <complex.h>

#include <cblas.h>
extern "C" {
#include <lapack.h>
}

namespace gt
{

namespace blas
{

// ======================================================================
// types aliases

using index_t = int;

using stream_t = dummy_stream;

// ======================================================================
// handle and stream management

class dummy_handle
{};

class handle_host : public detail::handle_base<handle_host, dummy_handle>
{
public:
  handle_host() {}
  ~handle_host() {}

  void set_stream(gt::stream_view sview) {}

  gt::stream_view get_stream() { return gt::stream_view{}; }
};

using handle_t = handle_host;

// ======================================================================
// axpy

template <typename T>
inline void axpy(handle_t& h, int n, T a, const T* x, int incx, T* y, int incy);

#define CREATE_AXPY(METHOD, GTTYPE, BLASTYPE)                                  \
  template <>                                                                  \
  inline void axpy<GTTYPE>(handle_t & h, int n, GTTYPE a, const GTTYPE* x,     \
                           int incx, GTTYPE* y, int incy)                      \
  {                                                                            \
    METHOD(n, a, reinterpret_cast<const BLASTYPE*>(x), incx,                   \
           reinterpret_cast<BLASTYPE*>(y), incy);                              \
  }

#define CREATE_AXPY_CMPLX(METHOD, GTTYPE, BLASTYPE)                            \
  template <>                                                                  \
  inline void axpy<GTTYPE>(handle_t & h, int n, GTTYPE a, const GTTYPE* x,     \
                           int incx, GTTYPE* y, int incy)                      \
  {                                                                            \
    METHOD(n, reinterpret_cast<BLASTYPE*>(&a),                                 \
           reinterpret_cast<const BLASTYPE*>(x), incx,                         \
           reinterpret_cast<BLASTYPE*>(y), incy);                              \
  }

CREATE_AXPY_CMPLX(cblas_zaxpy, gt::complex<double>, openblas_complex_double)
CREATE_AXPY_CMPLX(cblas_caxpy, gt::complex<float>, openblas_complex_float)
CREATE_AXPY(cblas_daxpy, double, double)
CREATE_AXPY(cblas_saxpy, float, float)

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
    METHOD(n, fac, reinterpret_cast<BLASTYPE*>(arr), incx);                    \
  }

#define CREATE_SCAL_CMPLX(METHOD, GTTYPE, BLASTYPE)                            \
  template <>                                                                  \
  inline void scal<GTTYPE, GTTYPE>(handle_t & h, int n, GTTYPE fac,            \
                                   GTTYPE* arr, const int incx)                \
  {                                                                            \
    METHOD(n, reinterpret_cast<BLASTYPE*>(&fac),                               \
           reinterpret_cast<BLASTYPE*>(arr), incx);                            \
  }

CREATE_SCAL_CMPLX(cblas_zscal, gt::complex<double>, openblas_complex_double)
CREATE_SCAL_CMPLX(cblas_cscal, gt::complex<float>, openblas_complex_float)
CREATE_SCAL(cblas_dscal, double, double)
CREATE_SCAL(cblas_sscal, float, float)

#undef CREATE_SCAL

// ======================================================================
// (zd|cs)scal

template <>
inline void scal<double, gt::complex<double>>(handle_t& h, int n, double fac,
                                              gt::complex<double>* arr,
                                              const int incx)
{
  cblas_zdscal(n, fac, reinterpret_cast<openblas_complex_double*>(arr), incx);
}

template <>
inline void scal<float, gt::complex<float>>(handle_t& h, int n, float fac,
                                            gt::complex<float>* arr,
                                            const int incx)
{
  cblas_csscal(n, fac, reinterpret_cast<openblas_complex_float*>(arr), incx);
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
    METHOD(n, reinterpret_cast<const BLASTYPE*>(x), incx,                      \
           reinterpret_cast<BLASTYPE*>(y), incy);                              \
  }

CREATE_COPY(cblas_zcopy, gt::complex<double>, openblas_complex_double)
CREATE_COPY(cblas_ccopy, gt::complex<float>, openblas_complex_float)
CREATE_COPY(cblas_dcopy, double, double)
CREATE_COPY(cblas_scopy, float, float)

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
    return METHOD(n, reinterpret_cast<const BLASTYPE*>(x), incx,               \
                  reinterpret_cast<const BLASTYPE*>(y), incy);                 \
  }

CREATE_DOT(cblas_ddot, double, double)
CREATE_DOT(cblas_sdot, float, float)

#undef CREATE_DOT

template <typename T>
inline T dotu(handle_t& h, int n, const T* x, int incx, const T* y, int incy);

#define CREATE_DOTU(METHOD, GTTYPE, BLASTYPE, RPART, IPART)                    \
  template <>                                                                  \
  inline GTTYPE dotu<GTTYPE>(handle_t & h, int n, const GTTYPE* x, int incx,   \
                             const GTTYPE* y, int incy)                        \
  {                                                                            \
    BLASTYPE result = METHOD(n, reinterpret_cast<const BLASTYPE*>(x), incx,    \
                             reinterpret_cast<const BLASTYPE*>(y), incy);      \
    return {RPART(result), IPART(result)};                                     \
  }

CREATE_DOTU(cblas_zdotu, gt::complex<double>, openblas_complex_double, creal,
            cimag)
CREATE_DOTU(cblas_cdotu, gt::complex<float>, openblas_complex_float, crealf,
            cimagf)

#undef CREATE_DOTU

template <typename T>
inline T dotc(handle_t& h, int n, const T* x, int incx, const T* y, int incy);

#define CREATE_DOTC(METHOD, GTTYPE, BLASTYPE, RPART, IPART)                    \
  template <>                                                                  \
  inline GTTYPE dotc<GTTYPE>(handle_t & h, int n, const GTTYPE* x, int incx,   \
                             const GTTYPE* y, int incy)                        \
  {                                                                            \
    BLASTYPE result = METHOD(n, reinterpret_cast<const BLASTYPE*>(x), incx,    \
                             reinterpret_cast<const BLASTYPE*>(y), incy);      \
    return {RPART(result), IPART(result)};                                     \
  }

CREATE_DOTC(cblas_zdotc, gt::complex<double>, openblas_complex_double, creal,
            cimag)
CREATE_DOTC(cblas_cdotc, gt::complex<float>, openblas_complex_float, crealf,
            cimagf)

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
    METHOD(CblasColMajor, CblasNoTrans, m, n, alpha,                           \
           reinterpret_cast<const BLASTYPE*>(A), lda,                          \
           reinterpret_cast<const BLASTYPE*>(x), incx, beta,                   \
           reinterpret_cast<BLASTYPE*>(y), incy);                              \
  }

#define CREATE_GEMV_CMPLX(METHOD, GTTYPE, BLASTYPE)                            \
  template <>                                                                  \
  inline void gemv<GTTYPE>(handle_t & h, int m, int n, GTTYPE alpha,           \
                           const GTTYPE* A, int lda, const GTTYPE* x,          \
                           int incx, GTTYPE beta, GTTYPE* y, int incy)         \
  {                                                                            \
    METHOD(CblasColMajor, CblasNoTrans, m, n,                                  \
           reinterpret_cast<const BLASTYPE*>(&alpha),                          \
           reinterpret_cast<const BLASTYPE*>(A), lda,                          \
           reinterpret_cast<const BLASTYPE*>(x), incx,                         \
           reinterpret_cast<BLASTYPE*>(&beta), reinterpret_cast<BLASTYPE*>(y), \
           incy);                                                              \
  }

CREATE_GEMV_CMPLX(cblas_zgemv, gt::complex<double>, openblas_complex_double)
CREATE_GEMV_CMPLX(cblas_cgemv, gt::complex<float>, openblas_complex_float)
CREATE_GEMV(cblas_dgemv, double, double)
CREATE_GEMV(cblas_sgemv, float, float)

#undef CREATE_GEMV

// ======================================================================
// getrf batched

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
    for (int b = 0; b < batchSize; b++) {                                      \
      METHOD(&n, &n, reinterpret_cast<BLASTYPE*>(d_Aarray[b]), &lda,           \
             &d_PivotArray[b * n], &d_infoArray[b]);                           \
    }                                                                          \
  }

CREATE_GETRF_BATCHED(LAPACK_zgetrf, gt::complex<double>, _Complex double)
CREATE_GETRF_BATCHED(LAPACK_cgetrf, gt::complex<float>, _Complex float)
CREATE_GETRF_BATCHED(LAPACK_dgetrf, double, double)
CREATE_GETRF_BATCHED(LAPACK_sgetrf, float, float)

#undef CREATE_GETRF_BATCHED

#if 0
// ======================================================================
// getrf_npvt batched

template <typename T>
inline void getrf_npvt_batched(handle_t& h, int n, T** d_Aarray, int lda,
                               int* d_infoArray, int batchSize);

#define CREATE_GETRF_NPVT_BATCHED(METHOD, GTTYPE, BLASTYPE)                    \
  template <>                                                                  \
  inline void getrf_npvt_batched<GTTYPE>(handle_t & h, int n,                  \
                                         GTTYPE** d_Aarray, int lda,           \
                                         int* d_infoArray, int batchSize)      \
  {                                                                            \
    for (int b = 0; b < batchSize; b++) {                                      \
      METHOD(&n, &n, reinterpret_cast<BLASTYPE*>(d_Aarray[b]), &lda, NULL,     \
             &d_infoArray[b]);                                                 \
    }                                                                          \
  }

CREATE_GETRF_NPVT_BATCHED(LAPACK_zgetrf, gt::complex<double>, _Complex double)
CREATE_GETRF_NPVT_BATCHED(LAPACK_cgetrf, gt::complex<float>, _Complex float)

#undef CREATE_GETRF_NPVT_BATCHED
#endif

// ======================================================================
// getrs batched

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
    static const char op_N = 'N';                                              \
    int info;                                                                  \
    for (int b = 0; b < batchSize; b++) {                                      \
      METHOD(&op_N, &n, &nrhs, reinterpret_cast<BLASTYPE*>(d_Aarray[b]), &lda, \
             &devIpiv[b * n], reinterpret_cast<BLASTYPE*>(d_Barray[b]), &ldb,  \
             &info);                                                           \
      if (info != 0) {                                                         \
        fprintf(stderr, "METHOD failed, info=%d at %s %d\n", info, __FILE__,   \
                __LINE__);                                                     \
        abort();                                                               \
      }                                                                        \
    }                                                                          \
  }

CREATE_GETRS_BATCHED(LAPACK_zgetrs, gt::complex<double>, _Complex double)
CREATE_GETRS_BATCHED(LAPACK_cgetrs, gt::complex<float>, _Complex float)
CREATE_GETRS_BATCHED(LAPACK_dgetrs, double, double)
CREATE_GETRS_BATCHED(LAPACK_sgetrs, float, float)

#undef CREATE_GETRS_BATCHED

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
    int lwork = n * n;                                                         \
    auto work = gt::empty<GTTYPE>({lwork});                                    \
    for (int b = 0; b < batchSize; b++) {                                      \
      for (int i = 0; i < n * n; i++) {                                        \
        d_Carray[b][i] = d_Aarray[b][i];                                       \
      }                                                                        \
      METHOD(&n, reinterpret_cast<BLASTYPE*>(d_Carray[b]), &lda,               \
             &devIpiv[b * n], reinterpret_cast<BLASTYPE*>(work.data()),        \
             &lwork, &d_infoArray[b]);                                         \
    }                                                                          \
  }

CREATE_GETRI_BATCHED(LAPACK_zgetri, gt::complex<double>, _Complex double)
CREATE_GETRI_BATCHED(LAPACK_cgetri, gt::complex<float>, _Complex float)
CREATE_GETRI_BATCHED(LAPACK_dgetri, double, double)
CREATE_GETRI_BATCHED(LAPACK_sgetri, float, float)

#undef CREATE_GETRI_BATCHED

// ======================================================================
// gemm batched

template <typename T>
inline void gemm_batched(handle_t& h, int m, int n, int k, T alpha,
                         T** d_Aarray, int lda, T** d_Barray, int ldb, T beta,
                         T** d_Carray, int ldc, int batchSize);

#define CREATE_GEMM_BATCHED_CMPLX(METHOD, GTTYPE, BLASTYPE)                    \
  template <>                                                                  \
  inline void gemm_batched<GTTYPE>(handle_t & h, int m, int n, int k,          \
                                   GTTYPE alpha, GTTYPE** d_Aarray, int lda,   \
                                   GTTYPE** d_Barray, int ldb, GTTYPE beta,    \
                                   GTTYPE** d_Carray, int ldc, int batchSize)  \
  {                                                                            \
    for (int b = 0; b < batchSize; b++) {                                      \
      METHOD(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k,               \
             reinterpret_cast<BLASTYPE*>(&alpha),                              \
             reinterpret_cast<BLASTYPE*>(d_Aarray[b]), lda,                    \
             reinterpret_cast<BLASTYPE*>(d_Barray[b]), ldb,                    \
             reinterpret_cast<BLASTYPE*>(&beta),                               \
             reinterpret_cast<BLASTYPE*>(d_Carray[b]), ldc);                   \
    }                                                                          \
  }

#define CREATE_GEMM_BATCHED(METHOD, GTTYPE, BLASTYPE)                          \
  template <>                                                                  \
  inline void gemm_batched<GTTYPE>(handle_t & h, int m, int n, int k,          \
                                   GTTYPE alpha, GTTYPE** d_Aarray, int lda,   \
                                   GTTYPE** d_Barray, int ldb, GTTYPE beta,    \
                                   GTTYPE** d_Carray, int ldc, int batchSize)  \
  {                                                                            \
    for (int b = 0; b < batchSize; b++) {                                      \
      METHOD(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha,        \
             reinterpret_cast<BLASTYPE*>(d_Aarray[b]), lda,                    \
             reinterpret_cast<BLASTYPE*>(d_Barray[b]), ldb, beta,              \
             reinterpret_cast<BLASTYPE*>(d_Carray[b]), ldc);                   \
    }                                                                          \
  }

CREATE_GEMM_BATCHED_CMPLX(cblas_zgemm, gt::complex<double>,
                          openblas_complex_double)
CREATE_GEMM_BATCHED_CMPLX(cblas_cgemm, gt::complex<float>,
                          openblas_complex_float)
CREATE_GEMM_BATCHED(cblas_dgemm, double, double);
CREATE_GEMM_BATCHED(cblas_sgemm, float, float);

#undef CREATE_GEMM_BATCHED

} // namespace blas

} // namespace gt

#endif
