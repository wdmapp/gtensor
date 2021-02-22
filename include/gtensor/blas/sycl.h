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

template <typename T>
inline void scal(handle_t* h, int n, T a, T* x, const int incx)
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
  T result;
  auto e = oneapi::mkl::blas::dot(*(h->handle), n, x, incx, y, incy, &result);
  e.wait();
  return result;
}

template <typename R>
inline gt::complex<R> dotu(handle_t* h, int n, const gt::complex<R>* x,
                           int incx, const gt::complex<R>* y, int incy)
{
  gt::complex<R> result;
  auto e = oneapi::mkl::blas::dotu(*(h->handle), n, x, incx, y, incy, &result);
  e.wait();
  return result;
}

template <typename R>
inline gt::complex<R> dotc(handle_t* h, int n, const gt::complex<R>* x,
                           int incx, const gt::complex<R>* y, int incy)
{
  gt::complex<R> result;
  auto e = oneapi::mkl::blas::dotc(*(h->handle), n, x, incx, y, incy, &result);
  e.wait();
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

  auto scratch_count = oneapi::mkl::lapack::getrf_batch_scratchpad_size<T>(
    q, n, n, lda, n * n, n, batchSize);
  auto scratch = sycl::malloc_device<T>(scratch_count, q);

  // NB: MKL expects a single contiguous array for the batch, as a host
  // pointer to device memory. Assume linear starting with the first
  // pointer, and copy it back to the host.
  T* d_Aptr;
  auto memcpy_e = q.memcpy(&d_Aptr, d_Aarray, sizeof(T*));
  memcpy_e.wait();

  auto e =
    oneapi::mkl::lapack::getrf_batch(q, n, n, d_Aptr, lda, n * n, d_PivotArray,
                                     n, batchSize, scratch, scratch_count);
  e.wait();

  sycl::free(scratch, q);
}

template <typename T>
inline void getrs_batched(handle_t* h, int n, int nrhs, T* const* d_Aarray,
                          int lda, gt::blas::index_t* devIpiv, T** d_Barray,
                          int ldb, int batchSize)
{
  sycl::queue& q = *(h->handle);

  auto scratch_count = oneapi::mkl::lapack::getrs_batch_scratchpad_size<T>(
    q, oneapi::mkl::transpose::nontrans, n, nrhs, lda, n * n, n, ldb, n * nrhs,
    batchSize);
  auto scratch = sycl::malloc_device<T>(scratch_count, q);

  // NB: MKL expects a single contiguous array for the batch, as a host
  // pointer to device memory. Assume linear starting with the first
  // pointer, and copy it back to the host.
  T* d_Aptr;
  T* d_Bptr;
  auto memcpy_A = q.memcpy(&d_Aptr, d_Aarray, sizeof(T*));
  memcpy_A.wait();
  auto memcpy_B = q.memcpy(&d_Bptr, d_Barray, sizeof(T*));
  memcpy_B.wait();

  auto e = oneapi::mkl::lapack::getrs_batch(
    q, oneapi::mkl::transpose::nontrans, n, nrhs, d_Aptr, lda, n * n, devIpiv,
    n, d_Bptr, ldb, n * nrhs, batchSize, scratch, scratch_count);
  e.wait();

  sycl::free(scratch, q);
}

} // namespace blas

} // namespace gt

#endif
