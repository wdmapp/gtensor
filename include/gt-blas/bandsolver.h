#ifndef GTENSOR_BANDSOLVE_H
#define GTENSOR_BANDSOLVE_H

#include "gtensor/complex.h"
#include "gtensor/reductions.h"

#include "gt-blas/blas.h"

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
 * @param h gt::blas::handle_t object
 * @param n size of each A_i
 * @param d_Aarray Array of device pointers to input [A_i]
 * @param lda leading distance of each A_i, >=n
 * @param batchSize number of matrices [A_i] and [B_i] in batch
 * @return matrix_bandwidth struct containing max upper and lower bandwidth
 */
template <typename T>
inline matrix_bandwidth get_max_bandwidth(handle_t& h, int n, T** d_Aarray,
                                          int lda, int batch_size)
{
  matrix_bandwidth res;

  auto d_batch_lbw = gt::empty_device<int>({batch_size});
  auto d_batch_ubw = gt::empty_device<int>({batch_size});

  auto launch_shape = gt::shape(batch_size);
  auto k_batch_lbw = d_batch_lbw.to_kernel();
  auto k_batch_ubw = d_batch_ubw.to_kernel();

  auto stream = h.get_stream();

  gt::launch<1>(
    launch_shape,
    GT_LAMBDA(int batch) {
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
    },
    stream);

  res.lower = gt::max(d_batch_lbw, stream);
  res.upper = gt::max(d_batch_ubw, stream);

  return res;
}

/**
 * Solve a batch of banded square LU factored matrices and RHS vectors.
 *
 * @see gt::blas::getrf_batched()
 * @see gt::blas::get_max_bandwidth()
 *
 * @param h gt::blas::handle_t object
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
inline void getrs_banded_batched(handle_t& h, int n, int nrhs, T** d_Aarray,
                                 int lda, index_t* d_PivotArray, T** d_Barray,
                                 int ldb, int batchSize, int lbw, int ubw)
{
#ifdef GTENSOR_DEVICE_SYCL
  sycl::queue& q = h.get_backend_handle();
  constexpr size_t blockSize = 32u;
  auto range =
    sycl::range<3>{static_cast<size_t>(nrhs), static_cast<size_t>(batchSize),
                   static_cast<size_t>(blockSize)};
  auto work_group_size = sycl::range<3>{1, 1, blockSize};
  const auto local_mem_size =
    q.get_device().get_info<sycl::info::device::local_mem_size>();
  if (n <= local_mem_size / sizeof(T)) {
#if __INTEL_CLANG_COMPILER < 20230000
    using local_accessor_t =
      sycl::accessor<T, 1, access::mode::read_write, access::target::local>;
#else
    using local_accessor_t = sycl::local_accessor<T, 1>;
#endif
    q.submit([&](sycl::handler& h) {
      auto sol = local_accessor_t({static_cast<size_t>(n)}, h);
      h.parallel_for(
        sycl::nd_range{range, work_group_size},
        [=](sycl::nd_item<3> idx) [[sycl::reqd_sub_group_size(blockSize)]] {
          auto rhs = idx.get_global_id(0);
          auto batch = idx.get_global_id(1);
          int o = idx.get_global_id(2);
          auto sg = idx.get_sub_group();
          T* A = d_Aarray[batch];
          T* B = d_Barray[batch] + ldb * rhs;
          index_t* piv = d_PivotArray + batch * n;
          T tmp;

          for (int b = o; b < n; b += blockSize) {
            sol[b] = B[b];
          }
          sycl::group_barrier(sg);

          if (o == 0) {
            for (int i = 0; i < n; i++) {
              std::swap(sol[i], sol[piv[i] - 1]);
            }
          }
          sycl::group_barrier(sg);

          for (int i = 0; i < n; ++i) {
            T x = sol[i];
            for (int b = i + o + 1; b <= i + lbw && b < n; b += blockSize) {
              sol[b] -= A[i * lda + b] * x;
            }
            sycl::group_barrier(sg);
          }

          for (int i = n - 1; i >= 0; --i) {
            T x = sol[i] / A[i * lda + i];
            int b = i + o - ubw;
            for (; b < i; b += blockSize) {
              if (b >= 0) {
                sol[b] -= A[i * lda + b] * x;
              }
            }
            sycl::group_barrier(sg);
            if (o == 0) {
              sol[i] = x;
            }
          }

          sycl::group_barrier(sg);
          for (int b = o; b < n; b += blockSize) {
            B[b] = sol[b];
          }
          sycl::group_barrier(sg);
        });
    });
  } else {
    q.submit([&](sycl::handler& h) {
      h.parallel_for(
        sycl::nd_range{range, work_group_size},
        [=](sycl::nd_item<3> idx) [[sycl::reqd_sub_group_size(blockSize)]] {
          auto rhs = idx.get_global_id(0);
          auto batch = idx.get_global_id(1);
          int o = idx.get_global_id(2);
          auto sg = idx.get_sub_group();
          T* A = d_Aarray[batch];
          T* B = d_Barray[batch] + ldb * rhs;
          index_t* piv = d_PivotArray + batch * n;
          T tmp;

          if (o == 0) {
            for (int i = 0; i < n; i++) {
              std::swap(B[i], B[piv[i] - 1]);
            }
          }
          sycl::group_barrier(sg);

          for (int i = 0; i < n; ++i) {
            T x = B[i];
            for (int b = i + o + 1; b <= i + lbw && b < n; b += blockSize) {
              B[b] -= A[i * lda + b] * x;
            }
            sycl::group_barrier(sg);
          }

          for (int i = n - 1; i >= 0; --i) {
            T x = B[i] / A[i * lda + i];
            int b = i + o - ubw;
            for (; b < i; b += blockSize) {
              if (b >= 0) {
                B[b] -= A[i * lda + b] * x;
              }
            }
            sycl::group_barrier(sg);
            if (o == 0) {
              B[i] = x;
            }
          }
        });
    });
  }
#else
  auto stream = h.get_stream();
  auto launch_shape = gt::shape(nrhs, batchSize);
  gt::launch<2>(
    launch_shape,
    GT_LAMBDA(int rhs, int batch) {
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
    },
    stream);
#endif
}

/**
 * Invert a batch of square LU factored matrices.
 *
 * @see gt::blas::getrf_batched()
 * @see gt::blas::get_max_bandwidth()
 *
 * @param h gt::blas::handle_t object
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
inline void invert_banded_batched(handle_t& h, int n, T** d_Aarray, int lda,
                                  index_t* d_PivotArray, T** d_Barray, int ldb,
                                  int batchSize, int lbw, int ubw)
{
  int nrhs = n;
  auto stream = h.get_stream();
  auto launch_shape = gt::shape(nrhs, batchSize);
  gt::launch<2>(
    launch_shape,
    GT_LAMBDA(int rhs, int batch) {
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
    },
    stream);
}

/**
 * Naive batched matrix multiply for solving with inverted matrices C = A^-1 * B
 *
 * DEPRECATED
 *
 * All matrices must be col-major
 *
 * @see gt::blas::get_max_bandwidth()
 * @see gt::blas::getrf_batched()
 * @see gt::blas::invert_banded_batched()
 *
 * @param h gt::blas::handle_t object
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
inline void solve_inverted_batched(handle_t& h, int n, int nrhs, T** d_Aarray,
                                   int lda, T** d_Barray, int ldb, T** d_Carray,
                                   int ldc, int batchSize)
{
  auto launch_shape = gt::shape(nrhs, batchSize);
  auto stream = h.get_stream();
  gt::launch<2>(
    launch_shape,
    GT_LAMBDA(int rhs, int batch) {
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
    },
    stream);
}

} // namespace blas

} // namespace gt

#endif // GTENSOR_BANDSOLVE_H
