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

  gt::backend::device_storage<T> d_rp(1);
  T result;
  auto e = oneapi::mkl::blas::dot(*(h->handle), n, x, incx, y, incy,
                                  gt::raw_pointer_cast(d_rp.data()));
  e.wait();
  auto e2 = q.memcpy(&result, gt::raw_pointer_cast(d_rp.data()), sizeof(T));
  e2.wait();
  return result;
}

template <typename R>
inline gt::complex<R> dotu(handle_t* h, int n, const gt::complex<R>* x,
                           int incx, const gt::complex<R>* y, int incy)
{
  sycl::queue& q = *(h->handle);
  using T = gt::complex<R>;

  gt::backend::device_storage<T> d_rp(1);
  T result;
  auto e = oneapi::mkl::blas::dotu(*(h->handle), n, x, incx, y, incy,
                                   gt::raw_pointer_cast(d_rp.data()));
  e.wait();
  auto e2 = q.memcpy(&result, gt::raw_pointer_cast(d_rp.data()), sizeof(T));
  e2.wait();
  return result;
}

template <typename R>
inline gt::complex<R> dotc(handle_t* h, int n, const gt::complex<R>* x,
                           int incx, const gt::complex<R>* y, int incy)
{
  sycl::queue& q = *(h->handle);
  using T = gt::complex<R>;

  gt::backend::device_storage<T> d_rp(1);
  T result;
  auto e = oneapi::mkl::blas::dotc(*(h->handle), n, x, incx, y, incy,
                                   gt::raw_pointer_cast(d_rp.data()));
  e.wait();
  auto e2 = q.memcpy(&result, gt::raw_pointer_cast(d_rp.data()), sizeof(T));
  e2.wait();
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

/*
namespace detail
{
template <typename T, typename S = space::sycl_managed>
using managed_allocator =
  typename backend::allocator_impl::selector<T, S>::type;

template <typename T>
using managed_storage =
  backend::gtensor_storage<T, managed_allocator<T>, space::sycl_managed>;
} // end namespace detail
*/

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
  /*
  using pivot_alloc_shared_t =
    sycl::usm_allocator<index_t*, sycl::usm::alloc::shared>;
  pivot_alloc_shared_t pivot_alloc(q);
  std::vector<index_t*, pivot_alloc_shared_t> d_PivotPtr(batchSize,
                                                         pivot_alloc);
                                                         */
  gt::backend::managed_storage<index_t*> d_PivotPtr(batchSize);
  for (int i = 0; i < batchSize; i++) {
    d_PivotPtr[i] = d_PivotArray + (i * n);
  }

  auto scratch_count = oneapi::mkl::lapack::getrf_batch_scratchpad_size<T>(
    q, &n64, &n64, &lda64, 1, &batchSize64);
  gt::backend::device_storage<T> scratch(scratch_count);

  auto e = oneapi::mkl::lapack::getrf_batch(
    q, &n64, &n64, d_Aarray, &lda64, gt::raw_pointer_cast(d_PivotPtr.data()), 1,
    &batchSize64, gt::raw_pointer_cast(scratch.data()), scratch_count);
  e.wait();

  // set zero to indicate no errors, which is true if we get here without
  // an exception being thrown
  // TODO: translate exceptions to info error codes?
  auto e2 = q.memset(d_infoArray, 0, sizeof(int) * batchSize);
  e2.wait();
}

template <typename T>
inline void getrf_npvt_batched(handle_t* h, int n, T** d_Aarray, int lda,
                               int* d_infoArray, int batchSize)
{
  sycl::queue& q = *(h->handle);

  auto scratch_count =
    oneapi::mkl::lapack::getrfnp_batch_strided_scratchpad_size<T>(q, n, n, lda,
                                                                  1, batchSize);
  gt::backend::device_storage<T> scratch(scratch_count);

  // TODO: hack until getrfnp group API is available, this won't work
  // for non-contiguous batches
  auto e = oneapi::mkl::lapack::getrfnp_batch_strided(
    q, n, n, d_Aarray[0], lda, 1, batchSize,
    gt::raw_pointer_cast(scratch.data()), scratch_count);
  e.wait();

  // set zero to indicate no errors, which is true if we get here without
  // an exception being thrown
  // TODO: translate exceptions to info error codes?
  auto e2 = q.memset(d_infoArray, 0, sizeof(int) * batchSize);
  e2.wait();
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
  /*
  using pivot_alloc_shared_t =
    sycl::usm_allocator<index_t*, sycl::usm::alloc::shared>;
  pivot_alloc_shared_t pivot_alloc(q);
  std::vector<index_t*, pivot_alloc_shared_t> d_PivotPtr(batchSize,
                                                         pivot_alloc);
  */

  gt::backend::managed_storage<index_t*> d_PivotPtr(batchSize);
  for (int i = 0; i < batchSize; i++) {
    d_PivotPtr[i] = d_PivotArray + (i * n);
  }

  auto trans_op = oneapi::mkl::transpose::nontrans;
  auto scratch_count = oneapi::mkl::lapack::getrs_batch_scratchpad_size<T>(
    q, &trans_op, &n64, &nrhs64, &lda64, &ldb64, 1, &batchSize64);
  gt::backend::device_storage<T> scratch(scratch_count);

  auto e = oneapi::mkl::lapack::getrs_batch(
    q, &trans_op, &n64, &nrhs64, d_Aarray, &lda64,
    gt::raw_pointer_cast(d_PivotPtr.data()), d_Barray, &ldb64, 1, &batchSize64,
    gt::raw_pointer_cast(scratch.data()), scratch_count);
  e.wait();
}

} // namespace blas

} // namespace gt

#endif
