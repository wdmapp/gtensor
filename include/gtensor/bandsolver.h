#ifndef GTENSOR_BANDSOLVE_H
#define GTENSOR_BANDSOLVE_H

#include "gtensor/gtensor.h"
#include "gtensor/reductions.h"

namespace gt
{

namespace bandsolver
{

#ifdef GTENSOR_DEVICE_SYCL
using index_t = std::int64_t;
#else
using index_t = std::int32_t;
#endif

struct matrix_bandwidth
{
  int lower;
  int upper;
};

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

} // namespace bandsolver

} // namespace gt

#endif // GTENSOR_BANDSOLVE_H
