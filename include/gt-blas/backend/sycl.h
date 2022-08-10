#ifndef GTENSOR_BLAS_HIP_H
#define GTENSOR_BLAS_HIP_H

#include "CL/sycl.hpp"
#include "oneapi/mkl.hpp"

namespace gt
{

namespace blas
{

// ======================================================================
// types aliases

using stream_t = cl::sycl::queue*;
using index_t = std::int64_t;

// ======================================================================
// handle and stream management

class handle_sycl : public detail::handle_base<handle_sycl, cl::sycl::queue>
{
public:
  handle_sycl() { handle_ = gt::backend::sycl::get_queue(); }

  void set_stream(gt::stream_view sview)
  {
    handle_ = sview.get_backend_stream();
  }

  gt::stream_view get_stream() { return gt::stream_view{handle_}; }
};

using handle_t = handle_sycl;

// ======================================================================
// axpy

template <typename T>
inline void axpy(handle_t& h, int n, T a, const T* x, int incx, T* y, int incy)
{
  auto e =
    oneapi::mkl::blas::axpy(h.get_backend_handle(), n, a, x, incx, y, incy);
  e.wait();
}

// ======================================================================
// scal

template <typename S, typename T>
inline void scal(handle_t& h, int n, S a, T* x, const int incx)
{
  auto e = oneapi::mkl::blas::scal(h.get_backend_handle(), n, a, x, incx);
  e.wait();
}

// ======================================================================
// copy

template <typename T>
inline void copy(handle_t& h, int n, const T* x, int incx, T* y, int incy)
{
  auto e = oneapi::mkl::blas::copy(h.get_backend_handle(), n, x, incx, y, incy);
  e.wait();
}

// ======================================================================
// dot, dotc (conjugate)

template <typename T>
inline T dot(handle_t& h, int n, const T* x, int incx, const T* y, int incy)
{
  sycl::queue& q = h.get_backend_handle();

  gt::space::device_vector<T> d_rp(1);
  T result;
  auto e = oneapi::mkl::blas::dot(q, n, x, incx, y, incy,
                                  gt::raw_pointer_cast(d_rp.data()));
  e.wait();
  auto e2 = q.memcpy(&result, gt::raw_pointer_cast(d_rp.data()), sizeof(T));
  e2.wait();
  return result;
}

template <typename R>
inline gt::complex<R> dotu(handle_t& h, int n, const gt::complex<R>* x,
                           int incx, const gt::complex<R>* y, int incy)
{
  sycl::queue& q = h.get_backend_handle();
  using T = gt::complex<R>;

  gt::space::device_vector<T> d_rp(1);
  T result;
  auto e = oneapi::mkl::blas::dotu(q, n, x, incx, y, incy,
                                   gt::raw_pointer_cast(d_rp.data()));
  e.wait();
  auto e2 = q.memcpy(&result, gt::raw_pointer_cast(d_rp.data()), sizeof(T));
  e2.wait();
  return result;
}

template <typename R>
inline gt::complex<R> dotc(handle_t& h, int n, const gt::complex<R>* x,
                           int incx, const gt::complex<R>* y, int incy)
{
  sycl::queue& q = h.get_backend_handle();
  using T = gt::complex<R>;

  gt::space::device_vector<T> d_rp(1);
  T result;
  auto e = oneapi::mkl::blas::dotc(q, n, x, incx, y, incy,
                                   gt::raw_pointer_cast(d_rp.data()));
  e.wait();
  auto e2 = q.memcpy(&result, gt::raw_pointer_cast(d_rp.data()), sizeof(T));
  e2.wait();
  return result;
}

// ======================================================================
// gemv

template <typename T>
inline void gemv(handle_t& h, int m, int n, T alpha, const T* A, int lda,
                 const T* x, int incx, T beta, T* y, int incy)
{
  auto e = oneapi::mkl::blas::gemv(h.get_backend_handle(),
                                   oneapi::mkl::transpose::nontrans, m, n,
                                   alpha, A, lda, x, incx, beta, y, incy);
  e.wait();
}

// ======================================================================
// getrf/getrs batched

template <typename T>
inline void getrf_batched(handle_t& h, int n, T** d_Aarray, int lda,
                          gt::blas::index_t* d_PivotArray, int* d_infoArray,
                          int batchSize)
{
  sycl::queue& q = h.get_backend_handle();

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
inline void getrf_npvt_batched(handle_t& h, int n, T** d_Aarray, int lda,
                               int* d_infoArray, int batchSize)
{
  sycl::queue& q = h.get_backend_handle();

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
inline void getrs_batched(handle_t& h, int n, int nrhs, T** d_Aarray, int lda,
                          gt::blas::index_t* d_PivotArray, T** d_Barray,
                          int ldb, int batchSize)
{
  sycl::queue& q = h.get_backend_handle();

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
inline void getri_batched(handle_t& h, int n, T** d_Aarray, int lda,
                          gt::blas::index_t* d_PivotArray, T** d_Carray,
                          int ldc, int* d_infoArray, int batchSize)
{
  sycl::queue& q = h.get_backend_handle();

  index_t n64 = n;
  index_t lda64 = lda;
  index_t ldc64 = ldc;
  index_t batchSize64 = batchSize;

  assert(lda == ldc);

  // unlike cuBLAS/rocBLAS, the pivot array to getri is expected to be
  // an array of pointer, just like d_Aarray.
  gt::space::managed_vector<index_t*> d_PivotPtr(batchSize);
  for (int i = 0; i < batchSize; i++) {
    d_PivotPtr[i] = d_PivotArray + (i * n);
  }

  // Note: cuBLAS API is out of place; we mimic that here by copying factored A
  // to C and inverting C in place. This might be more efficient if using a
  // separate out of place queue.
  gt::space::host_vector<T*> h_Aarray(batchSize);
  gt::space::host_vector<T*> h_Carray(batchSize);
  q.copy(d_Aarray, gt::raw_pointer_cast(h_Aarray.data()), batchSize);
  q.copy(d_Carray, gt::raw_pointer_cast(h_Carray.data()), batchSize);
  q.wait();

  for (index_t i = 0; i < batchSize; i++) {
    q.copy(h_Aarray[i], h_Carray[i], n * n);
  }
  q.wait();

  auto scratch_count = oneapi::mkl::lapack::getri_batch_scratchpad_size<T>(
    q, &n64, &lda64, 1, &batchSize64);
  gt::space::device_vector<T> scratch(scratch_count);

  auto e = oneapi::mkl::lapack::getri_batch(
    q, &n64, d_Carray, &ldc64, gt::raw_pointer_cast(d_PivotPtr.data()), 1,
    &batchSize64, gt::raw_pointer_cast(scratch.data()), scratch_count);
  e.wait();
}

template <typename T>
inline void gemm_batched(handle_t& h, int m, int n, int k, T alpha,
                         T** d_Aarray, int lda, T** d_Barray, int ldb, T beta,
                         T* d_Carray[], int ldc, int batchSize)
{
  sycl::queue& q = h.get_backend_handle();

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

  // Note: the spans for alpha/beta can have one value per matrix,
  // or one for all (size 1). We use size 1 here since that is all the interface
  // supports. Using a device accessible memory allocation still appears to be
  // necessary in current MKL versions.

  // Use RIAA to make sure it's cleaned up, in pure SYCL
  sycl::usm_allocator<T, sycl::usm::alloc::shared> allocator(q);
  std::vector<T, decltype(allocator)> alphabeta(2, allocator);

  sycl::span salpha(alphabeta.data(), 1);
  sycl::span sbeta(alphabeta.data() + 1, 1);
  salpha[0] = alpha;
  sbeta[0] = beta;

  sycl::span<const T*> sA{const_cast<const T**>(d_Aarray), batchSize_size_t};
  sycl::span<const T*> sB{const_cast<const T**>(d_Barray), batchSize_size_t};
  sycl::span<T*> sC{d_Carray, batchSize_size_t};

  cl::sycl::event gemm_batch_done;

  gemm_batch_done = oneapi::mkl::blas::gemm_batch(
    q, strans, strans, sm, sn, sk, salpha, sA, slda, sB, sldb, sbeta, sC, sldc,
    1, sbatchSize);
  gemm_batch_done.wait();
}

} // namespace blas

} // namespace gt

#endif
