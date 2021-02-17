#include "gtensor/blas.h"
#include "gtensor/cblas.h"

static gt::blas::handle_t handle = 0;

void gtblas_create()
{
  if (handle == 0) {
    gt::blas::create(&handle);
  }
}

void gtblas_destroy()
{
  if (handle != 0) {
    gt::blas::destroy(handle);
    handle = 0;
  }
}

void gtblas_set_stream(gtblas_stream_t stream_id)
{
  gt::blas::set_stream(handle, stream_id);
}

void gtblas_get_stream(gtblas_stream_t* stream_id)
{
  gt::blas::get_stream(handle, stream_id);
}

#define CREATE_C_AXPY(CNAME, CTYPE, CPPTYPE)                                   \
  void CNAME(int n, const CTYPE* a, const CTYPE* x, int incx, CTYPE* y,        \
             int incy)                                                         \
  {                                                                            \
    gt::blas::axpy(handle, n, (CPPTYPE*)a, (CPPTYPE*)x, incx, (CPPTYPE*)y,     \
                   incy);                                                      \
  }

CREATE_C_AXPY(gtblas_saxpy, float, float)
CREATE_C_AXPY(gtblas_daxpy, double, double)
CREATE_C_AXPY(gtblas_caxpy, gtblas_complex_float_t, gt::complex<float>)
CREATE_C_AXPY(gtblas_zaxpy, gtblas_complex_double_t, gt::complex<double>)

#if 0

void gtblas_zdscal(int n, const double fac, gtblas_complex_double_t* arr,
                    const int incx)
{
#ifdef GTENSOR_DEVICE_CUDA
  gtGpuCheck((cudaError_t)cublasZdscal(handle, n, &fac, arr, incx));
#elif defined(GTENSOR_DEVICE_HIP)
  gtGpuCheck((hipError_t)rocblas_zdscal(handle, n, &fac, arr, incx));
#elif defined(GTENSOR_DEVICE_SYCL)
  // TODO: exception handling
  auto e = oneapi::mkl::blas::scal(*handle, n, fac, arr, incx);
  e.wait();
#endif
}

/* ------------ copy --------------- */
void gtblas_zcopy(int n, const gtblas_complex_double_t* x, int incx,
                   gtblas_complex_double_t* y, int incy)
{
#ifdef GTENSOR_DEVICE_CUDA
  gtGpuCheck((cudaError_t)cublasZcopy(handle, n, x, incx, y, incy));
#elif defined(GTENSOR_DEVICE_HIP)
  gtGpuCheck((hipError_t)rocblas_zcopy(handle, n, x, incx, y, incy));
#elif defined(GTENSOR_DEVICE_SYCL)
  // TODO: exception handling
  auto e = oneapi::mkl::blas::copy(*handle, n, x, incx, y, incy);
  e.wait();
#endif
}

/* ------------ gemv --------------- */
void gtblas_dgemv(int m, int n, const double* alpha, const double* A, int lda,
                   const double* x, int incx, const double* beta, double* y,
                   int incy)
{
#ifdef GTENSOR_DEVICE_CUDA
  gtGpuCheck((cudaError_t)cublasDgemv(handle, CUBLAS_OP_N, m, n, alpha, A, lda,
                                      x, incx, beta, y, incy));
#elif defined(GTENSOR_DEVICE_HIP)
  gtGpuCheck((hipError_t)rocblas_dgemv(handle, rocblas_operation_none, m, n,
                                       alpha, A, lda, x, incx, beta, y, incy));
#elif defined(GTENSOR_DEVICE_SYCL)
  // TODO: exception handling
  auto e = oneapi::mkl::blas::gemv(*handle, oneapi::mkl::transpose::nontrans, m,
                                   n, *alpha, A, lda, x, incx, *beta, y, incy);
  e.wait();
#endif
}

void gtblas_zgemv(int m, int n, const gtblas_complex_double_t* alpha,
                   const gtblas_complex_double_t* A, int lda,
                   const gtblas_complex_double_t* x, int incx,
                   const gtblas_complex_double_t* beta,
                   gtblas_complex_double_t* y, int incy)
{
#ifdef GTENSOR_DEVICE_CUDA
  gtGpuCheck((cudaError_t)cublasZgemv(handle, CUBLAS_OP_N, m, n, alpha, A, lda,
                                      x, incx, beta, y, incy));
#elif defined(GTENSOR_DEVICE_HIP)
  gtGpuCheck((hipError_t)rocblas_zgemv(handle, rocblas_operation_none, m, n,
                                       alpha, A, lda, x, incx, beta, y, incy));
#elif defined(GTENSOR_DEVICE_SYCL)
  // TODO: exception handling
  auto e = oneapi::mkl::blas::gemv(*handle, oneapi::mkl::transpose::nontrans, m,
                                   n, *alpha, A, lda, x, incx, *beta, y, incy);
  e.wait();
#endif
}

void gtblas_zgetrf_batched(int n, gtblas_complex_double_t** d_Aarray, int lda,
                            gtblas_index_t* d_PivotArray, int* d_infoArray,
                            int batchSize)
{
#ifdef GTENSOR_DEVICE_CUDA
  gtGpuCheck((cudaError_t)cublasZgetrfBatched(
    handle, n, d_Aarray, lda, d_PivotArray, d_infoArray, batchSize));
#elif defined(GTENSOR_DEVICE_HIP)
  // Note: extra args are for general n x m size, and strideP for the
  // pivot array stride (we use n).
  gtGpuCheck((hipError_t)rocsolver_zgetrf_batched(
    handle, n, n, d_Aarray, lda, d_PivotArray, n, d_infoArray, batchSize));
#elif defined(GTENSOR_DEVICE_SYCL)
  // TODO: exception handling
  auto scratch_count =
    oneapi::mkl::lapack::getrf_batch_scratchpad_size<gtblas_complex_double_t>(
      *handle, n, n, lda, n * n, n, batchSize);
  auto scratch =
    sycl::malloc_device<gtblas_complex_double_t>(scratch_count, *handle);

  // NB: MKL expects a single contiguous array for the batch, as a host
  // pointer to device memory. Assume linear starting with the first
  // pointer, and copy it back to the host.
  gtblas_complex_double_t* d_Aptr;
  auto memcpy_e =
    handle->memcpy(&d_Aptr, d_Aarray, sizeof(gtblas_complex_double_t*));
  memcpy_e.wait();

  auto e = oneapi::mkl::lapack::getrf_batch(*handle, n, n, d_Aptr, lda, n * n,
                                            d_PivotArray, n, batchSize, scratch,
                                            scratch_count);
  e.wait();

  sycl::free(scratch, *handle);
#endif
}

void gtblas_zgetrs_batched(int n, int nrhs,
                            gtblas_complex_double_t* const* d_Aarray, int lda,
                            gtblas_index_t* devIpiv,
                            gtblas_complex_double_t** d_Barray, int ldb,
                            int batchSize)
{
#ifdef GTENSOR_DEVICE_CUDA
  int info;
  gtGpuCheck((cudaError_t)cublasZgetrsBatched(handle, CUBLAS_OP_N, n, nrhs,
                                              d_Aarray, lda, devIpiv, d_Barray,
                                              ldb, &info, batchSize));
  if (info != 0) {
    fprintf(stderr, "cublasDgetrsBatched failed, info=%d at %s %d\n", info,
            __FILE__, __LINE__);
    abort();
  }
#elif defined(GTENSOR_DEVICE_HIP)
  gtGpuCheck((hipError_t)rocsolver_zgetrs_batched(
    handle, rocblas_operation_none, n, nrhs, d_Aarray, lda, devIpiv, n,
    d_Barray, ldb, batchSize));
#elif defined(GTENSOR_DEVICE_SYCL)
  // TODO: exception handling
  auto scratch_count =
    oneapi::mkl::lapack::getrs_batch_scratchpad_size<gtblas_complex_double_t>(
      *handle, oneapi::mkl::transpose::nontrans, n, nrhs, lda, n * n, n, ldb,
      n * nrhs, batchSize);
  auto scratch =
    sycl::malloc_device<gtblas_complex_double_t>(scratch_count, *handle);

  // NB: MKL expects a single contiguous array for the batch, as a host
  // pointer to device memory. Assume linear starting with the first
  // pointer, and copy it back to the host.
  gtblas_complex_double_t* d_Aptr;
  gtblas_complex_double_t* d_Bptr;
  auto memcpy_A =
    handle->memcpy(&d_Aptr, d_Aarray, sizeof(gtblas_complex_double_t*));
  memcpy_A.wait();
  auto memcpy_B =
    handle->memcpy(&d_Bptr, d_Barray, sizeof(gtblas_complex_double_t*));
  memcpy_B.wait();

  auto e = oneapi::mkl::lapack::getrs_batch(
    *handle, oneapi::mkl::transpose::nontrans, n, nrhs, d_Aptr, lda, n * n,
    devIpiv, n, d_Bptr, ldb, n * nrhs, batchSize, scratch, scratch_count);
  e.wait();

  sycl::free(scratch, *handle);
#endif
}

void gtblas_dgetrf_batched(int n, double** d_Aarray, int lda,
                            gtblas_index_t* d_PivotArray, int* d_infoArray,
                            int batchSize)
{
#ifdef GTENSOR_DEVICE_CUDA
  gtGpuCheck((cudaError_t)cublasDgetrfBatched(
    handle, n, d_Aarray, lda, d_PivotArray, d_infoArray, batchSize));
#elif defined(GTENSOR_DEVICE_HIP)
  // Note: extra args are for general n x m size, and strideP for the
  // pivot array stride (we use n).
  gtGpuCheck((hipError_t)rocsolver_dgetrf_batched(
    handle, n, n, d_Aarray, lda, d_PivotArray, n, d_infoArray, batchSize));
#elif defined(GTENSOR_DEVICE_SYCL)
  // TODO: exception handling
  auto scratch_count = oneapi::mkl::lapack::getrf_batch_scratchpad_size<double>(
    *handle, n, n, lda, n * n, n, batchSize);

  auto scratch = sycl::malloc_device<double>(scratch_count, *handle);

  // NB: MKL expects a single contiguous array for the batch, as a host
  // pointer to device memory. Assume linear starting with the first
  // pointer, and copy it back to the host.
  double* d_Aptr;
  auto memcpy_e = handle->memcpy(&d_Aptr, d_Aarray, sizeof(double*));
  memcpy_e.wait();

  auto e = oneapi::mkl::lapack::getrf_batch(*handle, n, n, d_Aptr, lda, n * n,
                                            d_PivotArray, n, batchSize, scratch,
                                            scratch_count);
  e.wait();

  sycl::free(scratch, *handle);
#endif
}

void gtblas_dgetrs_batched(int n, int nrhs, double* const* d_Aarray, int lda,
                            gtblas_index_t* devIpiv, double** d_Barray,
                            int ldb, int batchSize)
{
#ifdef GTENSOR_DEVICE_CUDA
  int info;
  gtGpuCheck((cudaError_t)cublasDgetrsBatched(handle, CUBLAS_OP_N, n, nrhs,
                                              d_Aarray, lda, devIpiv, d_Barray,
                                              ldb, &info, batchSize));
  if (info != 0) {
    fprintf(stderr, "cublasDgetrsBatched failed, info=%d at %s %d\n", info,
            __FILE__, __LINE__);
    abort();
  }
#elif defined(GTENSOR_DEVICE_HIP)
  gtGpuCheck((hipError_t)rocsolver_dgetrs_batched(
    handle, rocblas_operation_none, n, nrhs, d_Aarray, lda, devIpiv, n,
    d_Barray, ldb, batchSize));
#elif defined(GTENSOR_DEVICE_SYCL)
  oneapi::mkl::transpose t = oneapi::mkl::transpose::nontrans;
  auto scratch_count = oneapi::mkl::lapack::getrs_batch_scratchpad_size<double>(
    *handle, t, n, nrhs, lda, n * n, n, ldb, n, batchSize);
  auto scratch = sycl::malloc_device<double>(scratch_count, *handle);

  // NB: MKL expects a single contiguous array for the batch, as a host
  // pointer to device memory. Assume linear starting with the first
  // pointer, and copy it back to the host.
  double* d_Aptr;
  double* d_Bptr;
  auto memcpy_A = handle->memcpy(&d_Aptr, d_Aarray, sizeof(double*));
  memcpy_A.wait();
  auto memcpy_B = handle->memcpy(&d_Bptr, d_Barray, sizeof(double*));
  memcpy_B.wait();

  try {
    auto e = oneapi::mkl::lapack::getrs_batch(
      *handle, t, n, nrhs, d_Aptr, lda, n * n, devIpiv, n, d_Bptr, ldb,
      n * nrhs, batchSize, scratch, scratch_count);
    e.wait();
  } catch (sycl::exception& e) {
    fprintf(stderr, "getrs_batch failed: %s (%s:%d)\n", e.what(), __FILE__,
            __LINE__);
    abort();
  }

  sycl::free(scratch, *handle);
#endif
}

#endif
