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
  if (stream_id == nullptr) {
    // set back to default stream / queue
    h->handle = &gt::backend::sycl::get_queue();
  } else {
    h->handle = stream_id;
  }
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

  gt::space::device_vector<T> d_rp(1);
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

  gt::space::device_vector<T> d_rp(1);
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

  gt::space::device_vector<T> d_rp(1);
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
  gt::space::managed_vector<index_t*> d_PivotPtr(batchSize);
  for (int i = 0; i < batchSize; i++) {
    d_PivotPtr[i] = d_PivotArray + (i * n);
  }

  auto scratch_count = oneapi::mkl::lapack::getrf_batch_scratchpad_size<T>(
    q, &n64, &n64, &lda64, 1, &batchSize64);
  gt::space::device_vector<T> scratch(scratch_count);

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

  // TODO: This uses the strides batch API, which only works when batch
  // data is contiguous. Replace when group batch API is available in oneMKL

  auto scratch_count = oneapi::mkl::lapack::getrfnp_batch_scratchpad_size<T>(
    q, n, n, lda, n * n, batchSize);
  gt::space::device_vector<T> scratch(scratch_count);

  // NB: check that input is contiguous, until the group API is available
  gt::space::host_vector<T*> h_Aarray(batchSize);
  q.copy(d_Aarray, h_Aarray.data(), batchSize).wait();
  for (int i = 0; i < batchSize - 1; i++) {
    assert(h_Aarray[i + 1] == h_Aarray[i] + n * n);
  }

  auto e = oneapi::mkl::lapack::getrfnp_batch(
    q, n, n, h_Aarray[0], lda, n * n, batchSize,
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
  gt::space::managed_vector<index_t*> d_PivotPtr(batchSize);
  for (int i = 0; i < batchSize; i++) {
    d_PivotPtr[i] = d_PivotArray + (i * n);
  }

  auto trans_op = oneapi::mkl::transpose::nontrans;
  auto scratch_count = oneapi::mkl::lapack::getrs_batch_scratchpad_size<T>(
    q, &trans_op, &n64, &nrhs64, &lda64, &ldb64, 1, &batchSize64);
  gt::space::device_vector<T> scratch(scratch_count);

  auto e = oneapi::mkl::lapack::getrs_batch(
    q, &trans_op, &n64, &nrhs64, d_Aarray, &lda64,
    gt::raw_pointer_cast(d_PivotPtr.data()), d_Barray, &ldb64, 1, &batchSize64,
    gt::raw_pointer_cast(scratch.data()), scratch_count);
  e.wait();
}

template <typename T>
inline void gemm_batched(handle_t* h, int m, int n, int k, T alpha,
                         T** d_Aarray, int lda, T** d_Barray, int ldb, T beta,
                         T* d_Carray[], int ldc, int batchSize)
{
  sycl::queue& q = *(h->handle);

  index_t m64 = m;
  index_t n64 = n;
  index_t k64 = k;
  index_t lda64 = lda;
  index_t ldb64 = ldb;
  index_t ldc64 = ldc;
  size_t batchSize_size_t = batchSize;

  // Note: one value per group
  sycl::span<index_t> sm{&m64, 1};
  sycl::span<index_t> sn{&n64, 1};
  sycl::span<index_t> sk{&k64, 1};
  sycl::span<index_t> slda{&lda64, 1};
  sycl::span<index_t> sldb{&ldb64, 1};
  sycl::span<index_t> sldc{&ldc64, 1};
  sycl::span<size_t> sbatchSize{&batchSize_size_t, 1};

  auto trans_op = oneapi::mkl::transpose::nontrans;
  sycl::span<oneapi::mkl::transpose> strans{&trans_op, 1};

  // Note: the arrays and alpha/beta have one value per matrix,
  // i.e. product batch sizes. With only one group, this is batchSize
  sycl::span salpha(sycl::malloc_shared<T>(batchSize_size_t, q),
                    batchSize_size_t);
  sycl::span sbeta(sycl::malloc_shared<T>(batchSize_size_t, q),
                   batchSize_size_t);

  for (int i = 0; i < batchSize; i++) {
    salpha[i] = alpha;
    sbeta[i] = beta;
  }

  sycl::span<const T*> sA{const_cast<const T**>(d_Aarray), batchSize_size_t};
  sycl::span<const T*> sB{const_cast<const T**>(d_Barray), batchSize_size_t};
  sycl::span<T*> sC{d_Carray, batchSize_size_t};

  cl::sycl::event gemm_batch_done;

  gemm_batch_done = oneapi::mkl::blas::gemm_batch(
    q, strans, strans, sm, sn, sk, salpha, sA, slda, sB, sldb, sbeta, sC, sldc,
    1, sbatchSize);
  gemm_batch_done.wait();

  sycl::free(salpha.data(), q);
  sycl::free(sbeta.data(), q);
}

} // namespace blas

} // namespace gt

#endif
