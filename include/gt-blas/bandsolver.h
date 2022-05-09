#ifndef GTENSOR_BANDSOLVE_H
#define GTENSOR_BANDSOLVE_H

#include "gtensor/complex.h"
#include "gtensor/reductions.h"

namespace gt
{

namespace blas
{

struct matrix_bandwidth
{
  int lower;
  int upper;
};

/**
 * Calculate max bandwidth in a batch of square matrices.
 *
 * @param n size of each A_i
 * @param d_Aarray Array of device pointers to input [A_i]
 * @param lda leading distance of each A_i, >=n
 * @param batchSize number of matrices [A_i] and [B_i] in batch
 * @return matrix_bandwidth struct containing max upper and lower bandwidth
 */
template <typename T>
inline matrix_bandwidth get_max_bandwidth(int n, T** d_Aarray, int lda,
                                          int batch_size)
{
  matrix_bandwidth res;

  auto d_batch_lbw = gt::empty_device<int>({batch_size});
  auto d_batch_ubw = gt::empty_device<int>({batch_size});

  auto launch_shape = gt::shape(batch_size);
  auto k_batch_lbw = d_batch_lbw.to_kernel();
  auto k_batch_ubw = d_batch_ubw.to_kernel();
  gt::launch<1>(
    launch_shape, GT_LAMBDA(int batch) {
      int lbw = n - 1;
      int ubw = n - 1;
      int i, j;
      T* A = d_Aarray[batch];

      // lower bandwdith, start at lower left
      bool all_zero = true;
      for (int d = 0; d < n - 1 && all_zero; d++) {
        i = n - 1;
        j = d;
        while (i >= 0 && j >= 0) {
          if (A[j * lda + i] != T(0)) {
            all_zero = false;
            break;
          }
          i--;
          j--;
        }
        if (all_zero) {
          lbw--;
        }
      }

      // upper bandwidth, start at top right
      all_zero = true;
      for (int d = n - 1; d > 0 && all_zero; d--) {
        i = 0;
        j = d;
        while (i < n && j < n) {
          if (A[j * lda + i] != T(0)) {
            all_zero = false;
            break;
          }
          i++;
          j++;
        }
        if (all_zero) {
          ubw--;
        }
      }

      k_batch_lbw(batch) = lbw;
      k_batch_ubw(batch) = ubw;
    });

  res.lower = gt::max(d_batch_lbw);
  res.upper = gt::max(d_batch_ubw);

  return res;
}

/**
 * Solve a batch of banded square LU factored matrices and RHS vectors.
 *
 * @see gt::blas::getrf_batched()
 * @see gt::blas::get_max_bandwidth()
 *
 * @param n size of each A_i and number of rows of each B_i
 * @param nrhs number of RHS column vectors in each B_i
 * @param d_Aarray Array of device pointers to LU factored input [A_i]
 * @param lda leading distance of each A_i, >=n
 * @param d_PivtoArray Array of device pointers to each piv_i
 * @param d_Barray Array of device pointers to each RHS/output [B_i]
 * @param ldb leading distance of each B_i, >=n
 * @param batchSize number of matrices [A_i] and [B_i] in batch
 * @param lbw max lower bandwidth of all [A_i]
 * @param ubw max upper bandwidth of all [A_i]
 */
template <typename T>
inline void getrs_banded_batched(int n, int nrhs, T** d_Aarray, int lda,
                                 index_t* d_PivotArray, T** d_Barray, int ldb,
                                 int batchSize, int lbw, int ubw)
{
  auto launch_shape = gt::shape(nrhs, batchSize);
  gt::launch<2>(
    launch_shape, GT_LAMBDA(int rhs, int batch) {
      T* A = d_Aarray[batch];
      T* B = d_Barray[batch] + ldb * rhs;
      index_t* piv = d_PivotArray + batch * n;
      T tmp;

      for (int i = 0; i < n; i++) {
        tmp = B[i];
        B[i] = B[piv[i] - 1];
        B[piv[i] - 1] = tmp;
      }

      // forward sub, unit diag
      for (int i = 0; i < lbw; i++) {
        tmp = B[i];
        for (int j = 0; j < i; j++) {
          tmp -= A[j * lda + i] * B[j];
        }
        B[i] = tmp;
      }
      for (int i = lbw; i < n; i++) {
        tmp = B[i];
        for (int j = i - lbw; j < i; j++) {
          tmp -= A[j * lda + i] * B[j];
        }
        B[i] = tmp;
      }

      // backward sub
      for (int i = n - 1; i > n - ubw - 1; i--) {
        tmp = B[i];
        for (int j = i + 1; j < n; j++) {
          tmp -= A[j * lda + i] * B[j];
        }
        B[i] = tmp / A[i * lda + i];
      }
      for (int i = n - ubw - 1; i >= 0; i--) {
        tmp = B[i];
        for (int j = i + 1; j <= i + ubw; j++) {
          tmp -= A[j * lda + i] * B[j];
        }
        B[i] = tmp / A[i * lda + i];
      }
    });
}

/**
 * Invert a batch of square LU factored matrices.
 *
 * @see gt::blas::getrf_batched()
 * @see gt::blas::get_max_bandwidth()
 *
 * @param n size of each A_i and B_i matrix in batch
 * @param d_Aarray Array of device pointers to each input A_i
 * @param lda leading distance of each A_i, >=n
 * @param d_PivtoArray Array of device pointers to each piv_i
 * @param d_Barray Array of device pointers to each output B_i
 * @param ldb leading distance of each B_i, >=n
 * @param batchSize number of matrices [A_i] and [B_i] in batch
 * @param lbw max lower bandwidth of all [A_i]
 * @param ubw max upper bandwidth of all [A_i]
 */
template <typename T>
inline void invert_banded_batched(int n, T** d_Aarray, int lda,
                                  index_t* d_PivotArray, T** d_Barray, int ldb,
                                  int batchSize, int lbw, int ubw)
{
  int nrhs = n;
  auto launch_shape = gt::shape(nrhs, batchSize);
  gt::launch<2>(
    launch_shape, GT_LAMBDA(int rhs, int batch) {
      T* A = d_Aarray[batch];
      T* B = d_Barray[batch] + ldb * rhs;
      index_t* piv = d_PivotArray + batch * n;
      T tmp;

      for (int i = 0; i < n; i++) {
        B[i] = 0;
      }
      B[rhs] = T(1);

      for (int i = 0; i < n; i++) {
        tmp = B[i];
        B[i] = B[piv[i] - 1];
        B[piv[i] - 1] = tmp;
      }

      // forward sub, unit diag
      for (int i = 0; i < lbw; i++) {
        tmp = B[i];
        for (int j = 0; j < i; j++) {
          tmp -= A[j * lda + i] * B[j];
        }
        B[i] = tmp;
      }
      for (int i = lbw; i < n; i++) {
        tmp = B[i];
        for (int j = i - lbw; j < i; j++) {
          tmp -= A[j * lda + i] * B[j];
        }
        B[i] = tmp;
      }

      // backward sub
      for (int i = n - 1; i > n - ubw - 1; i--) {
        tmp = B[i];
        for (int j = i + 1; j < n; j++) {
          tmp -= A[j * lda + i] * B[j];
        }
        B[i] = tmp / A[i * lda + i];
      }
      for (int i = n - ubw - 1; i >= 0; i--) {
        tmp = B[i];
        for (int j = i + 1; j <= i + ubw; j++) {
          tmp -= A[j * lda + i] * B[j];
        }
        B[i] = tmp / A[i * lda + i];
      }
    });
}

/**
 * Naive batched matrix multiply for solving with inverted matrices C = A^-1 * B
 *
 * All matrices must be col-major
 *
 * @see gt::blas::get_max_bandwidth()
 * @see gt::blas::getrf_batched()
 * @see gt::blas::invert_banded_batched()
 *
 * @param n size of each A^-1_i and rows of each B_i and C_i
 * @param nrhs number of RHS column vectors in each B_i and C_i
 * @param d_Aarray Array of device pointers to each inverted input A_i
 * @param lda leading distance of each A^-1_i, >=n
 * @param d_Barray Array of device pointers to each RHS input B_i
 * @param ldb leading distance of each B_i, >=n
 * @param d_Carray Array of device pointers to each output C_i
 * @param ldc leading distance of each C_i, >=n
 * @param batchSize number of matrices [A^-1_i], [B_i], [C_i] in batch
 */
template <typename T>
inline void solve_inverted_batched(int n, int nrhs, T** d_Aarray, int lda,
                                   T** d_Barray, int ldb, T** d_Carray, int ldc,
                                   int batchSize)
{
  auto launch_shape = gt::shape(nrhs, batchSize);
  gt::launch<2>(
    launch_shape, GT_LAMBDA(int rhs, int batch) {
      T* A = d_Aarray[batch];
      T* b = d_Barray[batch] + ldb * rhs;
      T* c = d_Carray[batch] + ldc * rhs;
      T* arow = nullptr; // track first element of row
      T tmp;

      // naive serial matrix-vector multiply (col major)
      for (int i = 0; i < n; i++) {
        tmp = T(0);
        arow = A + i;
        for (int k = 0; k < n; k++) {
          tmp += arow[k * lda] * b[k];
        }
        c[i] = tmp;
      }
    });
}

} // namespace blas

} // namespace gt

#endif // GTENSOR_BANDSOLVE_H
