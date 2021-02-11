#include "gtensor/gpublas.h"

#include <cstdio>
#include <cstdlib>

static gpublas_handle_t handle;

void gpublas_create()
{
  if (handle==0) {
    //std::cout<<"Create gpublas handle."<<std::endl;
#ifdef GTENSOR_DEVICE_CUDA
  gtGpuCheck((cudaError_t)cublasCreate(&handle));
#elif defined(GTENSOR_DEVICE_HIP)
  gtGpuCheck((hipError_t)rocblas_create_handle(&handle));
#elif defined(GTENSOR_DEVICE_SYCL)
  handle = &gt::backend::sycl::get_queue();
#endif
}
}

void gpublas_destroy()
{
  if (handle != 0 ) {
    //std::cout<<"Destroy gpublas handle."<<std::endl;
#ifdef GTENSOR_DEVICE_CUDA
  gtGpuCheck((cudaError_t)cublasDestroy(handle));
#elif defined(GTENSOR_DEVICE_HIP)
  gtGpuCheck((hipError_t)rocblas_destroy_handle(handle));
#endif
    handle=0;
  }
}

void gpublas_set_stream(gpublas_stream_t stream_id)
{
#ifdef GTENSOR_DEVICE_CUDA
  gtGpuCheck((cudaError_t)cublasSetStream(handle, stream_id));
#elif defined(GTENSOR_DEVICE_HIP)
  gtGpuCheck((hipError_t)rocblas_set_stream(handle, stream_id));
#elif defined(GTENSOR_DEVICE_SYCL)
  handle = stream_id;
#endif
}

void gpublas_get_stream(gpublas_stream_t* stream_id)
{
#ifdef GTENSOR_DEVICE_CUDA
  gtGpuCheck((cudaError_t)cublasGetStream(handle, stream_id));
#elif defined(GTENSOR_DEVICE_HIP)
  gtGpuCheck((hipError_t)rocblas_get_stream(handle, stream_id));
#elif defined(GTENSOR_DEVICE_SYCL)
  *stream_id = handle;
#endif
}

/* ---------------- axpy ------------------- */
void gpublas_zaxpy(int n, const gpublas_complex_double_t* a,
                   const gpublas_complex_double_t* x, int incx,
                   gpublas_complex_double_t* y, int incy)
{
#ifdef GTENSOR_DEVICE_CUDA
  gtGpuCheck((cudaError_t)cublasZaxpy(handle, n, a, x, incx, y, incy));
#elif defined(GTENSOR_DEVICE_HIP)
  gtGpuCheck((hipError_t)rocblas_zaxpy(handle, n, a, x, incx, y, incy));
#elif defined(GTENSOR_DEVICE_SYCL)
  // TODO: exception handling
  auto e = oneapi::mkl::blas::axpy(*handle, n, *a, x, incx, y, incy);
  e.wait();
#endif
}

void gpublas_daxpy(int n, const double* a, const double* x, int incx, double* y,
                   int incy)
{
#ifdef GTENSOR_DEVICE_CUDA
  gtGpuCheck((cudaError_t)cublasDaxpy(handle, n, a, x, incx, y, incy));
#elif defined(GTENSOR_DEVICE_HIP)
  gtGpuCheck((hipError_t)rocblas_daxpy(handle, n, a, x, incx, y, incy));
#elif defined(GTENSOR_DEVICE_SYCL)
  // TODO: exception handling
  auto e = oneapi::mkl::blas::axpy(*handle, n, *a, x, incx, y, incy);
  e.wait();
#endif
}

/*
void gpublas_caxpy(int n, const gt::complex<float> a,
                   const gt::complex<float>* x, int incx, gt::complex<float>* y,
                   int incy, int* status) {
#ifdef GTENSOR_DEVICE_CUDA
    gtGpuCheck((cudaError_t)cublasCaxpy(handle, n, a, x, incx, y, incy));
    cudaDeviceSynchronize();  // ???
#elif defined(GTENSOR_DEVICE_HIP)
    gtGpuCheck((cudaError_t)rocblas_caxpy(handle, n, a, x, incx, y, incy));
#elif defined(GTENSOR_DEVICE_SYCL)
    // TODO: exception handling
    auto e = oneapi::mkl::blas::axpy(*handle, n, a, x, incx, y, incy);
    e.wait();
#endif
}
*/

void gpublas_zdscal(int n, const double fac, gpublas_complex_double_t* arr,
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
void gpublas_zcopy(int n, const gpublas_complex_double_t* x, int incx,
                   gpublas_complex_double_t* y, int incy)
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
void gpublas_dgemv(int m, int n, const double* alpha, const double* A, int lda,
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

void gpublas_zgemv(int m, int n, const gpublas_complex_double_t* alpha,
		   const gpublas_complex_double_t* A, int lda,
                   const gpublas_complex_double_t* x, int incx,
		   const gpublas_complex_double_t* beta,
		   gpublas_complex_double_t* y, int incy)
{
#ifdef GTENSOR_DEVICE_CUDA
  gtGpuCheck((cudaError_t)cublasZgemv(handle, CUBLAS_OP_N, m, n, alpha, A, lda,
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

void gpublas_zgetrf_batched(int n, gpublas_complex_double_t** d_Aarray, int lda,
                            gpublas_index_t* d_PivotArray, int* d_infoArray,
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
    oneapi::mkl::lapack::getrf_batch_scratchpad_size<gpublas_complex_double_t>(
      *handle, n, n, lda, n * n, n, batchSize);
  auto scratch =
    sycl::malloc_device<gpublas_complex_double_t>(scratch_count, *handle);

  // NB: MKL expects a single contiguous array for the batch, as a host
  // pointer to device memory. Assume linear starting with the first
  // pointer, and copy it back to the host.
  gpublas_complex_double_t* d_Aptr;
  auto memcpy_e =
    handle->memcpy(&d_Aptr, d_Aarray, sizeof(gpublas_complex_double_t*));
  memcpy_e.wait();

  auto e = oneapi::mkl::lapack::getrf_batch(*handle, n, n, d_Aptr, lda, n * n,
                                            d_PivotArray, n, batchSize, scratch,
                                            scratch_count);
  e.wait();

  sycl::free(scratch, *handle);
#endif
}

void gpublas_zgetrs_batched(int n, int nrhs,
                            gpublas_complex_double_t* const* d_Aarray, int lda,
                            gpublas_index_t* devIpiv,
                            gpublas_complex_double_t** d_Barray, int ldb,
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
    oneapi::mkl::lapack::getrs_batch_scratchpad_size<gpublas_complex_double_t>(
      *handle, oneapi::mkl::transpose::nontrans, n, nrhs, lda, n * n, n, ldb,
      n * nrhs, batchSize);
  auto scratch =
    sycl::malloc_device<gpublas_complex_double_t>(scratch_count, *handle);

  // NB: MKL expects a single contiguous array for the batch, as a host
  // pointer to device memory. Assume linear starting with the first
  // pointer, and copy it back to the host.
  gpublas_complex_double_t* d_Aptr;
  gpublas_complex_double_t* d_Bptr;
  auto memcpy_A =
    handle->memcpy(&d_Aptr, d_Aarray, sizeof(gpublas_complex_double_t*));
  memcpy_A.wait();
  auto memcpy_B =
    handle->memcpy(&d_Bptr, d_Barray, sizeof(gpublas_complex_double_t*));
  memcpy_B.wait();

  auto e = oneapi::mkl::lapack::getrs_batch(
    *handle, oneapi::mkl::transpose::nontrans, n, nrhs, d_Aptr, lda, n * n,
    devIpiv, n, d_Bptr, ldb, n * nrhs, batchSize, scratch, scratch_count);
  e.wait();

  sycl::free(scratch, *handle);
#endif
}

void gpublas_dgetrf_batched(int n, double** d_Aarray, int lda,
                            gpublas_index_t* d_PivotArray, int* d_infoArray,
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

void gpublas_dgetrs_batched(int n, int nrhs, double* const* d_Aarray, int lda,
                            gpublas_index_t* devIpiv, double** d_Barray,
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
