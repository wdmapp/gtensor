#include "gtensor/blas.h"

#ifdef __cplusplus
extern "C" {
#endif

gt::blas::handle_t* gtblas_create();
void gtblas_destroy(gt::blas::handle_t* h);

void gtblas_set_stream(gt::blas::handle_t* h, gt::blas::stream_t stream_id);
void gtblas_get_stream(gt::blas::handle_t* h, gt::blas::stream_t* stream_id);

void gtblas_saxpy(gt::blas::handle_t* h, int n, float a, const float* x,
                  int incx, float* y, int incy);
void gtblas_daxpy(gt::blas::handle_t* h, int n, double a, const double* x,
                  int incx, double* y, int incy);
void gtblas_caxpy(gt::blas::handle_t* h, int n, gt::complex<float> a,
                  const gt::complex<float>* x, int incx, gt::complex<float>* y,
                  int incy);
void gtblas_zaxpy(gt::blas::handle_t* h, int n, gt::complex<double> a,
                  const gt::complex<double>* x, int incx,
                  gt::complex<double>* y, int incy);

void gtblas_sdscal(gt::blas::handle_t* h, int n, const float fac, float* arr,
                   const int incx);
void gtblas_ddscal(gt::blas::handle_t* h, int n, const double fac, double* arr,
                   const int incx);
void gtblas_cdscal(gt::blas::handle_t* h, int n, const float fac,
                   gt::complex<float>* arr, const int incx);
void gtblas_zdscal(gt::blas::handle_t* h, int n, const double fac,
                   gt::complex<double>* arr, const int incx);

void gtblas_scopy(gt::blas::handle_t* h, int n, const float* x, int incx,
                  float* y, int incy);
void gtblas_dcopy(gt::blas::handle_t* h, int n, const double* x, int incx,
                  double* y, int incy);
void gtblas_ccopy(gt::blas::handle_t* h, int n, const gt::complex<float>* x,
                  int incx, gt::complex<float>* y, int incy);
void gtblas_zcopy(gt::blas::handle_t* h, int n, const gt::complex<double>* x,
                  int incx, gt::complex<double>* y, int incy);

void gtblas_sdot(gt::blas::handle_t* h, int n, const float* x, int incx,
                 float* y, int incy);
void gtblas_ddot(gt::blas::handle_t* h, int n, const double* x, int incx,
                 double* y, int incy);
void gtblas_cdotu(gt::blas::handle_t* h, int n, const gt::complex<float>* x,
                  int incx, gt::complex<float>* y, int incy);
void gtblas_zdotu(gt::blas::handle_t* h, int n, const gt::complex<double>* x,
                  int incx, gt::complex<double>* y, int incy);
void gtblas_cdotc(gt::blas::handle_t* h, int n, const gt::complex<float>* x,
                  int incx, gt::complex<float>* y, int incy);
void gtblas_zdotc(gt::blas::handle_t* h, int n, const gt::complex<double>* x,
                  int incx, gt::complex<double>* y, int incy);

void gtblas_sgemv(gt::blas::handle_t* h, int m, int n, float alpha,
                  const float* A, int lda, const float* x, int incx, float beta,
                  float* y, int incy);
void gtblas_dgemv(gt::blas::handle_t* h, int m, int n, double alpha,
                  const double* A, int lda, const double* x, int incx,
                  double beta, double* y, int incy);
void gtblas_cgemv(gt::blas::handle_t* h, int m, int n, gt::complex<float> alpha,
                  const gt::complex<float>* A, int lda,
                  const gt::complex<float>* x, int incx,
                  gt::complex<float> beta, gt::complex<float>* y, int incy);
void gtblas_zgemv(gt::blas::handle_t* h, int m, int n,
                  gt::complex<double> alpha, const gt::complex<double>* A,
                  int lda, const gt::complex<double>* x, int incx,
                  gt::complex<double> beta, gt::complex<double>* y, int incy);

void gtblas_sgetrf_batched(gt::blas::handle_t* h, int n, float* d_Aarray[],
                           int lda, gt::blas::index_t* d_PivotArray,
                           int* d_infoArray, int batchSize);
void gtblas_sgetrs_batched(gt::blas::handle_t* h, int n, int nrhs,
                           float* const* d_Aarray, int lda,
                           gt::blas::index_t* devIpiv, float** d_Barray,
                           int ldb, int batchSize);
void gtblas_dgetrf_batched(gt::blas::handle_t* h, int n, double* d_Aarray[],
                           int lda, gt::blas::index_t* d_PivotArray,
                           int* d_infoArray, int batchSize);
void gtblas_dgetrs_batched(gt::blas::handle_t* h, int n, int nrhs,
                           double* const* d_Aarray, int lda,
                           gt::blas::index_t* devIpiv, double** d_Barray,
                           int ldb, int batchSize);
void gtblas_cgetrf_batched(gt::blas::handle_t* h, int n,
                           gt::complex<float>** d_Aarray, int lda,
                           gt::blas::index_t* d_PivotArray, int* d_infoArray,
                           int batchSize);
void gtblas_cgetrs_batched(gt::blas::handle_t* h, int n, int nrhs,
                           gt::complex<float>* const* d_Aarray, int lda,
                           gt::blas::index_t* devIpiv,
                           gt::complex<float>** d_Barray, int ldb,
                           int batchSize);
void gtblas_zgetrf_batched(gt::blas::handle_t* h, int n,
                           gt::complex<double>** d_Aarray, int lda,
                           gt::blas::index_t* d_PivotArray, int* d_infoArray,
                           int batchSize);
void gtblas_zgetrs_batched(gt::blas::handle_t* h, int n, int nrhs,
                           gt::complex<double>* const* d_Aarray, int lda,
                           gt::blas::index_t* devIpiv,
                           gt::complex<double>** d_Barray, int ldb,
                           int batchSize);

#ifdef __cplusplus
}
#endif
