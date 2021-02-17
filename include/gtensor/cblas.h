#include "gtensor/blas.h"

typedef gt::blas::handle_t gtblas_handle_t;
typedef gt::blas::stream_t gtblas_stream_t;
typedef gt::complex<double> gtblas_complex_double_t;
typedef gt::complex<float> gtblas_complex_float_t;
typedef int gtblas_index_t;

#ifdef __cplusplus
extern "C" {
#endif

void gtblas_create();
void gtblas_destroy();

void gtblas_set_stream(gtblas_stream_t stream_id);
void gtblas_get_stream(gtblas_stream_t* stream_id);

void gtblas_saxpy(int n, const float* a, const float* x, int incx, float* y,
                  int incy);
void gtblas_daxpy(int n, const double* a, const double* x, int incx, double* y,
                  int incy);
void gtblas_caxpy(int n, const gtblas_complex_float_t* a,
                  const gtblas_complex_float_t* x, int incx,
                  gtblas_complex_float_t* y, int incy);
void gtblas_zaxpy(int n, const gtblas_complex_double_t* a,
                  const gtblas_complex_double_t* x, int incx,
                  gtblas_complex_double_t* y, int incy);

#ifdef __cplusplus
}
#endif

#if 0
#ifdef GTENSOR_DEVICE_CUDA

#include "cublas_v2.h"
typedef cublasHandle_t gpublas_handle_t;
typedef cudaStream_t gpublas_stream_t;
typedef cuDoubleComplex gpublas_complex_double_t;
typedef cuComplex gpublas_complex_float_t;
typedef int gpublas_index_t;

#elif defined(GTENSOR_DEVICE_HIP)

#include "rocblas.h"
#include "rocsolver.h"
typedef rocblas_handle gpublas_handle_t;
typedef hipStream_t gpublas_stream_t;
typedef rocblas_double_complex gpublas_complex_double_t;
typedef rocblas_float_complex gpublas_complex_float_t;
typedef int gpublas_index_t;

#elif defined(GTENSOR_DEVICE_SYCL)

#include "CL/sycl.hpp"
#include "oneapi/mkl.hpp"
typedef cl::sycl::queue* gpublas_handle_t;
typedef cl::sycl::queue* gpublas_stream_t;

typedef gt::complex<double> gpublas_complex_double_t;
typedef gt::complex<float> gpublas_complex_float_t;
typedef std::int64_t gpublas_index_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

void gpublas_create();
void gpublas_destroy();

void gpublas_set_stream(gpublas_stream_t stream_id);
void gpublas_get_stream(gpublas_stream_t* stream_id);

void gpublas_zaxpy(int n, const gpublas_complex_double_t* a,
                   const gpublas_complex_double_t* x, int incx,
                   gpublas_complex_double_t* y, int incy);
void gpublas_daxpy(int n, const double* a, const double* x, int incx, double* y,
                   int incy);

void gpublas_zdscal(int n, const double fac, gpublas_complex_double_t* arr,
                    const int incx);

void gpublas_zcopy(int n, const gpublas_complex_double_t* x, int incx,
                   gpublas_complex_double_t* y, int incy);

void gpublas_dgemv(int m, int n, const double* alpha, const double* A, int lda,
                   const double* x, int incx, const double* beta, double* y,
                   int incy);

void gpublas_zgemv(int m, int n, const gpublas_complex_double_t* alpha,
                   const gpublas_complex_double_t* A, int lda,
                   const gpublas_complex_double_t* x, int incx,
                   const gpublas_complex_double_t* beta,
                   gpublas_complex_double_t* y, int incy);

void gpublas_dgetrf_batched(int n, double* d_Aarray[], int lda,
                            gpublas_index_t* d_PivotArray, int* d_infoArray,
                            int batchSize);
void gpublas_dgetrs_batched(int n, int nrhs, double* const* d_Aarray, int lda,
                            gpublas_index_t* devIpiv, double** d_Barray,
                            int ldb, int batchSize);
void gpublas_zgetrf_batched(int n, gpublas_complex_double_t** d_Aarray, int lda,
                            gpublas_index_t* d_PivotArray, int* d_infoArray,
                            int batchSize);
void gpublas_zgetrs_batched(int n, int nrhs,
                            gpublas_complex_double_t* const* d_Aarray, int lda,
                            gpublas_index_t* devIpiv,
                            gpublas_complex_double_t** d_Barray, int ldb,
                            int batchSize);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

namespace gt
{

namespace blas
{


template <typename T>
void axpy(int n, const T* a, const T* x, int incx, T* y, int incy);

template <typename T>
void scal(int n, const double fac, T* arr, const int incx);

template <typename T>
void copy(int n, const T* x, int incx, T* y, int incy);

template <typename T>
void gemv(int m, int n, const T* alpha, const double* A, int lda,
                 const T* x, int incx, const T* beta, T* y, int incy);

template <typename T>
void getrf_batched(int n, T* d_Aarray[], int lda,
                           gpublas_index_t* d_PivotArray, int* d_infoArray,
                           int batchSize);

template <typename T>
void getrs_batched(int n, int nrhs, T* const* d_Aarray, int lda,
                           gpublas_index_t* devIpiv, T** d_Barray,
                           int ldb, int batchSize);

}
}
#endif

#endif
