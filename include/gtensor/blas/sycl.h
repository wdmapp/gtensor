#ifndef GTENSOR_BLAS_HIP_H
#define GTENSOR_BLAS_HIP_H

#include "CL/sycl.hpp"
#include "oneapi/mkl.hpp"

namespace gt
{

namespace blas
{

struct handle_t
{
  cl::sycl::queue* handle;
};

// ======================================================================
// types aliases

using stream_t = cl::sycl::queue*;
using index_t = std::int64_t;

// ======================================================================
// handle and stream management

inline handle_t* create()
{
  handle_t* h = new handle_t();
  h->handle = &gt::backend::sycl::get_queue();
  return h;
}

inline void destroy(handle_t* h)
{
  h->handle = nullptr;
  delete h;
}

inline void set_stream(handle_t* h, stream_t stream_id)
{
  h->handle = stream_id;
}

inline void get_stream(handle_t* h, stream_t* stream_id)
{
  *stream_id = h->handle;
}

// ======================================================================
// axpy

template <typename T>
inline void axpy(handle_t* h, int n, T a, const T* x, int incx, T* y, int incy)
{
  auto e = oneapi::mkl::blas::axpy(*(h->handle), n, a, x, incx, y, incy);
  e.wait();
}

// ======================================================================
// scal

template <typename S, typename T>
inline void scal(handle_t* h, int n, S a, T* x, const int incx)
{
  auto e = oneapi::mkl::blas::scal(*(h->handle), n, a, x, incx);
  e.wait();
}

// ======================================================================
// copy

template <typename T>
inline void copy(handle_t* h, int n, const T* x, int incx, T* y, int incy)
{
  auto e = oneapi::mkl::blas::copy(*(h->handle), n, x, incx, y, incy);
  e.wait();
}

// ======================================================================
// dot, dotc (conjugate)

template <typename T>
inline T dot(handle_t* h, int n, const T* x, int incx, const T* y, int incy)
{
  sycl::queue& q = *(h->handle);

  T* d_rp = sycl::malloc_device<T>(1, q);
  T result;
  auto e = oneapi::mkl::blas::dot(*(h->handle), n, x, incx, y, incy, d_rp);
  e.wait();
  auto e2 = q.memcpy(&result, d_rp, sizeof(T));
  e2.wait();
  sycl::free(d_rp, q);
  return result;
}

template <typename R>
inline gt::complex<R> dotu(handle_t* h, int n, const gt::complex<R>* x,
                           int incx, const gt::complex<R>* y, int incy)
{
  sycl::queue& q = *(h->handle);
  using T = gt::complex<R>;

  T* d_rp = sycl::malloc_device<T>(1, q);
  T result;
  auto e = oneapi::mkl::blas::dotu(*(h->handle), n, x, incx, y, incy, d_rp);
  e.wait();
  auto e2 = q.memcpy(&result, d_rp, sizeof(T));
  e2.wait();
  sycl::free(d_rp, q);
  return result;
}

template <typename R>
inline gt::complex<R> dotc(handle_t* h, int n, const gt::complex<R>* x,
                           int incx, const gt::complex<R>* y, int incy)
{
  sycl::queue& q = *(h->handle);
  using T = gt::complex<R>;

  T* d_rp = sycl::malloc_device<T>(1, q);
  T result;
  auto e = oneapi::mkl::blas::dotc(*(h->handle), n, x, incx, y, incy, d_rp);
  e.wait();
  auto e2 = q.memcpy(&result, d_rp, sizeof(T));
  e2.wait();
  sycl::free(d_rp, q);
  return result;
}

// ======================================================================
// gemv

template <typename T>
inline void gemv(handle_t* h, int m, int n, T alpha, const T* A, int lda,
                 const T* x, int incx, T beta, T* y, int incy)
{
  auto e =
    oneapi::mkl::blas::gemv(*(h->handle), oneapi::mkl::transpose::nontrans, m,
                            n, alpha, A, lda, x, incx, beta, y, incy);
  e.wait();
}

// ======================================================================
// getrf/getrs batched

template <typename T>
inline void getrf_batched(handle_t* h, int n, T** d_Aarray, int lda,
                          gt::blas::index_t* d_PivotArray, int* d_infoArray,
                          int batchSize)
{
  sycl::queue& q = *(h->handle);

  index_t n64 = n;
  index_t lda64 = lda;
  index_t batchSize64 = batchSize;

  // unlike cuBLAS/rocBLAS, the pivot array to getrf is expected to be
  // an array of pointer, just like d_Aarray.
  auto d_PivotPtr = sycl::malloc_shared<index_t*>(batchSize, q);
  for (int i = 0; i < batchSize; i++) {
    d_PivotPtr[i] = d_PivotArray + (i * n);
  }

  auto scratch_count = oneapi::mkl::lapack::getrf_batch_scratchpad_size<T>(
    q, &n64, &n64, &lda64, 1, &batchSize64);

  auto scratch = sycl::malloc_device<T>(scratch_count, q);

  auto e = oneapi::mkl::lapack::getrf_batch(q, &n64, &n64, d_Aarray, &lda64,
                                            d_PivotPtr, 1, &batchSize64,
                                            scratch, scratch_count);
  // set zero to indicate no errors, which is true if we get here without
  // an exception being thrown
  // TODO: translate exceptions to info error codes?
  auto e2 = q.memset(d_infoArray, 0, sizeof(int) * batchSize);
  e.wait();
  e2.wait();

  sycl::free(scratch, q);
  sycl::free(d_PivotPtr, q);
}

template <typename T>
inline void getrs_batched(handle_t* h, int n, int nrhs, T** d_Aarray, int lda,
                          gt::blas::index_t* d_PivotArray, T** d_Barray,
                          int ldb, int batchSize)
{
  sycl::queue& q = *(h->handle);

  index_t n64 = n;
  index_t nrhs64 = nrhs;
  index_t lda64 = lda;
  index_t ldb64 = ldb;
  index_t batchSize64 = batchSize;

  // unlike cuBLAS/rocBLAS, the pivot array to getrf is expected to be
  // an array of pointer, just like d_Aarray.
  auto d_PivotPtr = sycl::malloc_shared<index_t*>(batchSize, q);
  for (int i = 0; i < batchSize; i++) {
    d_PivotPtr[i] = d_PivotArray + (i * n);
  }

  auto trans_op = oneapi::mkl::transpose::nontrans;
  auto scratch_count = oneapi::mkl::lapack::getrs_batch_scratchpad_size<T>(
    q, &trans_op, &n64, &nrhs64, &lda64, &ldb64, 1, &batchSize64);
  auto scratch = sycl::malloc_device<T>(scratch_count, q);

  auto e = oneapi::mkl::lapack::getrs_batch(
    q, &trans_op, &n64, &nrhs64, d_Aarray, &lda64, d_PivotPtr, d_Barray, &ldb64,
    1, &batchSize64, scratch, scratch_count);
  e.wait();

  sycl::free(scratch, q);
  sycl::free(d_PivotPtr, q);
}

} // namespace blas

} // namespace gt

#endif
