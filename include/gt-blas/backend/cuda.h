#ifndef GTENSOR_BLAS_CUDA_H
#define GTENSOR_BLAS_CUDA_H

#include "cublas_v2.h"

// ======================================================================
// error handling helper

#define gtBlasCheck(what)                                                      \
  {                                                                            \
    gtBlasCheckImpl(what, __FILE__, __LINE__);                                 \
  }

inline void gtBlasCheckImpl(cublasStatus_t code, const char* file, int line)
{
  if (code != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "gtBlasCheck: cublas status %d at %s:%d\n", code, file,
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

class handle_cuda : public detail::handle_base<handle_cuda, cublasHandle_t>
{
public:
  handle_cuda() { gtBlasCheck(cublasCreate(&handle_)); }
  ~handle_cuda() { gtBlasCheck(cublasDestroy(handle_)); }

  void set_stream(gt::stream_view sview)
  {
    gtBlasCheck(cublasSetStream(handle_, sview.get_backend_stream()));
  }

  gt::stream_view get_stream()
  {
    cudaStream_t s;
    gtBlasCheck(cublasGetStream(handle_, &s));
    return gt::stream_view{s};
  }
};

using handle_t = handle_cuda;

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

CREATE_AXPY(cublasZaxpy, gt::complex<double>, cuDoubleComplex)
CREATE_AXPY(cublasCaxpy, gt::complex<float>, cuComplex)
CREATE_AXPY(cublasDaxpy, double, double)
CREATE_AXPY(cublasSaxpy, float, float)

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

CREATE_SCAL(cublasZscal, gt::complex<double>, cuDoubleComplex)
CREATE_SCAL(cublasCscal, gt::complex<float>, cuComplex)
CREATE_SCAL(cublasDscal, double, double)
CREATE_SCAL(cublasSscal, float, float)

#undef CREATE_SCAL

// ======================================================================
// (zd|cs)scal

template <>
inline void scal<double, gt::complex<double>>(handle_t& h, int n, double fac,
                                              gt::complex<double>* arr,
                                              const int incx)
{
  gtBlasCheck(cublasZdscal(h.get_backend_handle(), n, &fac,
                           reinterpret_cast<cuDoubleComplex*>(arr), incx));
}

template <>
inline void scal<float, gt::complex<float>>(handle_t& h, int n, float fac,
                                            gt::complex<float>* arr,
                                            const int incx)
{
  gtBlasCheck(cublasCsscal(h.get_backend_handle(), n, &fac,
                           reinterpret_cast<cuComplex*>(arr), incx));
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

CREATE_COPY(cublasZcopy, gt::complex<double>, cuDoubleComplex)
CREATE_COPY(cublasCcopy, gt::complex<float>, cuComplex)
CREATE_COPY(cublasDcopy, double, double)
CREATE_COPY(cublasScopy, float, float)

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

CREATE_DOT(cublasDdot, double, double)
CREATE_DOT(cublasSdot, float, float)

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

CREATE_DOTU(cublasZdotu, gt::complex<double>, cuDoubleComplex)
CREATE_DOTU(cublasCdotu, gt::complex<float>, cuComplex)

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

CREATE_DOTC(cublasZdotc, gt::complex<double>, cuDoubleComplex)
CREATE_DOTC(cublasCdotc, gt::complex<float>, cuComplex)

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
    gtBlasCheck(METHOD(h.get_backend_handle(), CUBLAS_OP_N, m, n,              \
                       reinterpret_cast<BLASTYPE*>(&alpha),                    \
                       reinterpret_cast<const BLASTYPE*>(A), lda,              \
                       reinterpret_cast<const BLASTYPE*>(x), incx,             \
                       reinterpret_cast<BLASTYPE*>(&beta),                     \
                       reinterpret_cast<BLASTYPE*>(y), incy));                 \
  }

CREATE_GEMV(cublasZgemv, gt::complex<double>, cuDoubleComplex)
CREATE_GEMV(cublasCgemv, gt::complex<float>, cuComplex)
CREATE_GEMV(cublasDgemv, double, double)
CREATE_GEMV(cublasSgemv, float, float)

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
    gtBlasCheck(METHOD(h.get_backend_handle(), n,                              \
                       reinterpret_cast<BLASTYPE**>(d_Aarray), lda,            \
                       d_PivotArray, d_infoArray, batchSize));                 \
  }

CREATE_GETRF_BATCHED(cublasZgetrfBatched, gt::complex<double>, cuDoubleComplex)
CREATE_GETRF_BATCHED(cublasCgetrfBatched, gt::complex<float>, cuComplex)
CREATE_GETRF_BATCHED(cublasDgetrfBatched, double, double)
CREATE_GETRF_BATCHED(cublasSgetrfBatched, float, float)

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
    int info;                                                                  \
    gtBlasCheck(METHOD(h.get_backend_handle(), CUBLAS_OP_N, n, nrhs,           \
                       reinterpret_cast<BLASTYPE* const*>(d_Aarray), lda,      \
                       devIpiv, reinterpret_cast<BLASTYPE**>(d_Barray), ldb,   \
                       &info, batchSize));                                     \
    if (info != 0) {                                                           \
      fprintf(stderr, "METHOD failed, info=%d at %s %d\n", info, __FILE__,     \
              __LINE__);                                                       \
      abort();                                                                 \
    }                                                                          \
  }

CREATE_GETRS_BATCHED(cublasZgetrsBatched, gt::complex<double>, cuDoubleComplex)
CREATE_GETRS_BATCHED(cublasCgetrsBatched, gt::complex<float>, cuComplex)
CREATE_GETRS_BATCHED(cublasDgetrsBatched, double, double)
CREATE_GETRS_BATCHED(cublasSgetrsBatched, float, float)

#undef CREATE_GETRS_BATCHED

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
    gtBlasCheck(METHOD(h.get_backend_handle(), n,                              \
                       reinterpret_cast<BLASTYPE**>(d_Aarray), lda, NULL,      \
                       d_infoArray, batchSize));                               \
  }

CREATE_GETRF_NPVT_BATCHED(cublasZgetrfBatched, gt::complex<double>,
                          cuDoubleComplex)
CREATE_GETRF_NPVT_BATCHED(cublasCgetrfBatched, gt::complex<float>, cuComplex)

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
    gtBlasCheck(METHOD(h.get_backend_handle(), n,                              \
                       reinterpret_cast<BLASTYPE* const*>(d_Aarray), lda,      \
                       devIpiv, reinterpret_cast<BLASTYPE**>(d_Carray), ldc,   \
                       d_infoArray, batchSize));                               \
  }

CREATE_GETRI_BATCHED(cublasZgetriBatched, gt::complex<double>, cuDoubleComplex)
CREATE_GETRI_BATCHED(cublasCgetriBatched, gt::complex<float>, cuComplex)
CREATE_GETRI_BATCHED(cublasDgetriBatched, double, double)
CREATE_GETRI_BATCHED(cublasSgetriBatched, float, float)

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
    gtBlasCheck(METHOD(h.get_backend_handle(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, \
                       k, reinterpret_cast<BLASTYPE*>(&alpha),                 \
                       reinterpret_cast<BLASTYPE**>(d_Aarray), lda,            \
                       reinterpret_cast<BLASTYPE**>(d_Barray), ldb,            \
                       reinterpret_cast<BLASTYPE*>(&beta),                     \
                       reinterpret_cast<BLASTYPE**>(d_Carray), ldc,            \
                       batchSize));                                            \
  }

CREATE_GEMM_BATCHED(cublasZgemmBatched, gt::complex<double>, cuDoubleComplex)
CREATE_GEMM_BATCHED(cublasCgemmBatched, gt::complex<float>, cuComplex)
CREATE_GEMM_BATCHED(cublasDgemmBatched, double, double);
CREATE_GEMM_BATCHED(cublasSgemmBatched, float, float);

#undef CREATE_GEMM_BATCHED

} // namespace blas

} // namespace gt

#endif
