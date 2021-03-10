#include "gtensor/blas.h"

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
void gtblas_caxpy(int n, const gt::complex<float>* a,
                  const gt::complex<float>* x, int incx, gt::complex<float>* y,
                  int incy);
void gtblas_zaxpy(int n, const gt::complex<double>* a,
                  const gt::complex<double>* x, int incx,
                  gt::complex<double>* y, int incy);

void gtblas_sscal(int n, const float* fac, float* arr, int incx);
void gtblas_dscal(int n, const double* fac, double* arr, int incx);
void gtblas_cscal(int n, const gt::complex<float>* fac, gt::complex<float>* arr,
                  int incx);
void gtblas_zscal(int n, const gt::complex<double>* fac,
                  gt::complex<double>* arr, int incx);
void gtblas_zdscal(int n, const double* fac, gt::complex<double>* arr,
                   int incx);
void gtblas_csscal(int n, const float* fac, gt::complex<float>* arr, int incx);

void gtblas_scopy(int n, const float* x, int incx, float* y, int incy);
void gtblas_dcopy(int n, const double* x, int incx, double* y, int incy);
void gtblas_ccopy(int n, const gt::complex<float>* x, int incx,
                  gt::complex<float>* y, int incy);
void gtblas_zcopy(int n, const gt::complex<double>* x, int incx,
                  gt::complex<double>* y, int incy);

void gtblas_sdot(int n, const float* x, int incx, float* y, int incy);
void gtblas_ddot(int n, const double* x, int incx, double* y, int incy);
void gtblas_cdotu(int n, const gt::complex<float>* x, int incx,
                  gt::complex<float>* y, int incy);
void gtblas_zdotu(int n, const gt::complex<double>* x, int incx,
                  gt::complex<double>* y, int incy);
void gtblas_cdotc(int n, const gt::complex<float>* x, int incx,
                  gt::complex<float>* y, int incy);
void gtblas_zdotc(int n, const gt::complex<double>* x, int incx,
                  gt::complex<double>* y, int incy);

void gtblas_sgemv(int m, int n, const float* alpha, const float* A, int lda,
                  const float* x, int incx, const float* beta, float* y,
                  int incy);
void gtblas_dgemv(int m, int n, const double* alpha, const double* A, int lda,
                  const double* x, int incx, const double* beta, double* y,
                  int incy);
void gtblas_cgemv(int m, int n, const gt::complex<float>* alpha,
                  const gt::complex<float>* A, int lda,
                  const gt::complex<float>* x, int incx,
                  const gt::complex<float>* beta, gt::complex<float>* y,
                  int incy);
void gtblas_zgemv(int m, int n, const gt::complex<double>* alpha,
                  const gt::complex<double>* A, int lda,
                  const gt::complex<double>* x, int incx,
                  const gt::complex<double>* beta, gt::complex<double>* y,
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
void gtblas_cgetrf_batched(int n, gt::complex<float>** d_Aarray, int lda,
                           gt::blas::index_t* d_PivotArray, int* d_infoArray,
                           int batchSize);
void gtblas_cgetrs_batched(int n, int nrhs, gt::complex<float>** d_Aarray,
                           int lda, gt::blas::index_t* devIpiv,
                           gt::complex<float>** d_Barray, int ldb,
                           int batchSize);
void gtblas_zgetrf_batched(int n, gt::complex<double>** d_Aarray, int lda,
                           gt::blas::index_t* d_PivotArray, int* d_infoArray,
                           int batchSize);
void gtblas_zgetrs_batched(int n, int nrhs, gt::complex<double>** d_Aarray,
                           int lda, gt::blas::index_t* devIpiv,
                           gt::complex<double>** d_Barray, int ldb,
                           int batchSize);

#ifdef __cplusplus
}
#endif
