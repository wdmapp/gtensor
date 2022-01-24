#include "gt-blas/blas.h"

// Use 4/8 byte aligned type for Fortran interop; gt::complex may
// use thrust::complex which is 8/16 byte aligned
template <typename T>
using f2c_complex = std::complex<T>;

#ifdef __cplusplus
extern "C" {
#endif

void gtblas_create();
void gtblas_destroy();

void gtblas_set_stream(gt::blas::stream_t stream_id);
void gtblas_get_stream(gt::blas::stream_t* stream_id);

void gtblas_saxpy(int n, const float* a, const float* x, int incx, float* y,
                  int incy);
void gtblas_daxpy(int n, const double* a, const double* x, int incx, double* y,
                  int incy);
void gtblas_caxpy(int n, const f2c_complex<float>* a,
                  const f2c_complex<float>* x, int incx, f2c_complex<float>* y,
                  int incy);
void gtblas_zaxpy(int n, const f2c_complex<double>* a,
                  const f2c_complex<double>* x, int incx,
                  f2c_complex<double>* y, int incy);

void gtblas_sscal(int n, const float* fac, float* arr, int incx);
void gtblas_dscal(int n, const double* fac, double* arr, int incx);
void gtblas_cscal(int n, const f2c_complex<float>* fac, f2c_complex<float>* arr,
                  int incx);
void gtblas_zscal(int n, const f2c_complex<double>* fac,
                  f2c_complex<double>* arr, int incx);
void gtblas_zdscal(int n, const double* fac, f2c_complex<double>* arr,
                   int incx);
void gtblas_csscal(int n, const float* fac, f2c_complex<float>* arr, int incx);

void gtblas_scopy(int n, const float* x, int incx, float* y, int incy);
void gtblas_dcopy(int n, const double* x, int incx, double* y, int incy);
void gtblas_ccopy(int n, const f2c_complex<float>* x, int incx,
                  f2c_complex<float>* y, int incy);
void gtblas_zcopy(int n, const f2c_complex<double>* x, int incx,
                  f2c_complex<double>* y, int incy);

void gtblas_sdot(int n, const float* x, int incx, float* y, int incy);
void gtblas_ddot(int n, const double* x, int incx, double* y, int incy);
void gtblas_cdotu(int n, const f2c_complex<float>* x, int incx,
                  f2c_complex<float>* y, int incy);
void gtblas_zdotu(int n, const f2c_complex<double>* x, int incx,
                  f2c_complex<double>* y, int incy);
void gtblas_cdotc(int n, const f2c_complex<float>* x, int incx,
                  f2c_complex<float>* y, int incy);
void gtblas_zdotc(int n, const f2c_complex<double>* x, int incx,
                  f2c_complex<double>* y, int incy);

void gtblas_sgemv(int m, int n, const float* alpha, const float* A, int lda,
                  const float* x, int incx, const float* beta, float* y,
                  int incy);
void gtblas_dgemv(int m, int n, const double* alpha, const double* A, int lda,
                  const double* x, int incx, const double* beta, double* y,
                  int incy);
void gtblas_cgemv(int m, int n, const f2c_complex<float>* alpha,
                  const f2c_complex<float>* A, int lda,
                  const f2c_complex<float>* x, int incx,
                  const f2c_complex<float>* beta, f2c_complex<float>* y,
                  int incy);
void gtblas_zgemv(int m, int n, const f2c_complex<double>* alpha,
                  const f2c_complex<double>* A, int lda,
                  const f2c_complex<double>* x, int incx,
                  const f2c_complex<double>* beta, f2c_complex<double>* y,
                  int incy);

void gtblas_sgetrf_batched(int n, float** d_Aarray, int lda,
                           gt::blas::index_t* d_PivotArray, int* d_infoArray,
                           int batchSize);
void gtblas_sgetrs_batched(int n, int nrhs, float** d_Aarray, int lda,
                           gt::blas::index_t* devIpiv, float** d_Barray,
                           int ldb, int batchSize);
void gtblas_dgetrf_batched(int n, double** d_Aarray, int lda,
                           gt::blas::index_t* d_PivotArray, int* d_infoArray,
                           int batchSize);
void gtblas_dgetrs_batched(int n, int nrhs, double** d_Aarray, int lda,
                           gt::blas::index_t* devIpiv, double** d_Barray,
                           int ldb, int batchSize);
void gtblas_cgetrf_batched(int n, f2c_complex<float>** d_Aarray, int lda,
                           gt::blas::index_t* d_PivotArray, int* d_infoArray,
                           int batchSize);
void gtblas_cgetrs_batched(int n, int nrhs, f2c_complex<float>** d_Aarray,
                           int lda, gt::blas::index_t* devIpiv,
                           f2c_complex<float>** d_Barray, int ldb,
                           int batchSize);
void gtblas_zgetrf_batched(int n, f2c_complex<double>** d_Aarray, int lda,
                           gt::blas::index_t* d_PivotArray, int* d_infoArray,
                           int batchSize);
void gtblas_zgetrs_batched(int n, int nrhs, f2c_complex<double>** d_Aarray,
                           int lda, gt::blas::index_t* devIpiv,
                           f2c_complex<double>** d_Barray, int ldb,
                           int batchSize);
void gtblas_cgetrf_npvt_batched(int n, f2c_complex<float>** d_Aarray, int lda,
                                int* d_infoArray, int batchSize);
void gtblas_zgetrf_npvt_batched(int n, f2c_complex<double>** d_Aarray, int lda,
                                int* d_infoArray, int batchSize);

#ifdef __cplusplus
}
#endif
