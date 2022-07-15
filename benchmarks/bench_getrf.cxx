#include <iostream>
#include <numeric>
#include <string>

#include <benchmark/benchmark.h>

#include <gtensor/gtensor.h>

#include <gt-blas/blas.h>

#include "gt-bm.h"

using namespace gt::placeholders;

//#define BENCH_GETRF_DEBUG

template <typename Matrix>
void debug_print_matrix(const char* name, Matrix&& M)
{
#ifdef BENCH_GETRF_DEBUG
  int n = M.shape(0);
  std::cerr << name << "\n";
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < n; i++) {
      std::cerr << M(i, j) << " ";
    }
    std::cerr << "\n";
  }
  std::cerr << std::endl;
#endif
}

template <typename Vector>
void debug_print_vector(const char* name, Vector&& V)
{
#ifdef BENCH_GETRF_DEBUG
  int n = V.shape(0);
  std::cerr << name << "\n";
  for (int i = 0; i < n; i++) {
    std::cerr << V(i) << " ";
  }
  std::cerr << std::endl;
#endif
}

template <typename CT>
auto make_test_matrix(int n, int bw, int batch_size, bool needs_pivot)
{
  auto h_Adata = gt::zeros<CT>({n, n, batch_size});
  for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < n; i++) {
      h_Adata(i, i, b) = CT(bw + 1.0, 0.0);
      // set upper / lower diags at bw diagonal
      for (int d = 1; d <= bw; d++) {
        if (i + d < n) {
          h_Adata(i, i + d, b) = CT(-1.0, 0.0);
          h_Adata(i + d, i, b) = CT(0.0, -1.0);
        }
      }
    }
    if (needs_pivot) {
      h_Adata(0, 0, b) = CT(n / 64.0, 0);
    }
  }
  return h_Adata;
}

/**
 * Multiply the L and U factors stored in a single matrix, as returned by
 * BLAS getrf calls. The upper diaganol is U, and the lower diagonal is
 * L with an implicit diagonal of all ones (that otherwise would overlap
 * with U). Batched version takes a 3 dimensional gtensor, with
 * outermost dimension being the batch.
 *
 * Asynchronous, returns after kernel launch.
 *
 * This can be used to verify the correctness of the factoring.
 *
 * @param d_Aout device accessable 3d gtensor to store the result of
 *    multiplying L and U
 * @param d_LUin device accessable 3d gtensor containing batches of
 *    factored square matrices, as returned by getrf_batched
 * @param d_piv device accessable 2d gtensor containing pivot arrays
 *    for each batch, in getrs format (serial swap)
 * @param d_perm device accessable 2d gtensor of same size as d_piv
 *    used to convert the pivot array to index permutaiton format
 */
template <typename CT>
void lumm_batched(gt::gtensor_device<CT, 3>& d_Aout,
                  gt::gtensor_device<CT, 3>& d_LUin,
                  gt::gtensor_device<gt::blas::index_t, 2>& d_piv,
                  gt::gtensor_device<gt::blas::index_t, 2>& d_perm,
                  bool use_pivot)
{
  int n = d_LUin.shape(0);
  int nbatch = d_LUin.shape(2);
  auto launch_shape = gt::shape(nbatch);
  auto k_Aout = d_Aout.to_kernel();
  auto k_LUin = d_LUin.to_kernel();
  auto k_piv = d_piv.to_kernel();
  auto k_perm = d_perm.to_kernel();
  gt::launch<1>(
    launch_shape, GT_LAMBDA(int batch) {
      // Convert the row swap pivot into more convenient row permutation
      gt::blas::index_t itmp, imapped;
      for (gt::blas::index_t i = 0; i < n; i++) {
        k_perm(i, batch) = i;
      }

      if (use_pivot) {
        for (gt::blas::index_t i = 0; i < n; i++) {
          // piv is one based, arrays are zero based
          imapped = k_piv(i, batch) - 1;
          if (imapped != i) {
            itmp = k_perm(i, batch);
            k_perm(i, batch) = k_perm(imapped, batch);
            k_perm(imapped, batch) = itmp;
          }
        }
      }

      for (int j = 0; j < n; j++) {
        CT tmp;
        // handle diagonal of L being all ones implicitly, by avoiding
        // accessing index (i, i). For i <= j, this just means adding
        // 1 * U(i, j). Note that because of triangle structure,
        // both values are nonzero only when k <= min(i, j)
        for (int i = 0; i <= j; i++) {
          tmp = k_LUin(i, j, batch);
          for (int k = 0; k < i; k++) {
            tmp += k_LUin(i, k, batch) * k_LUin(k, j, batch);
          }
          k_Aout(k_perm(i, batch), j, batch) = tmp;
        }
        // for i > j, the index (i, i) is never reached so the base
        // value is 0
        for (int i = j + 1; i < n; i++) {
          tmp = CT(0);
          for (int k = 0; k <= j; k++) {
            tmp += k_LUin(i, k, batch) * k_LUin(k, j, batch);
          }
          k_Aout(k_perm(i, batch), j, batch) = tmp;
        }
      }
    });
}

template <typename T, typename CT = gt::complex<T>>
bool check_lu_batched(gt::gtensor<CT, 3>& h_Aactual,
                      gt::gtensor_device<CT, 3>& d_LU,
                      gt::gtensor_device<gt::blas::index_t, 2>& d_piv,
                      bool use_pivot)
{
  int n = h_Aactual.shape(0);
  int nbatch = h_Aactual.shape(2);
  constexpr T tol = 100.0 * std::numeric_limits<T>::epsilon();

  gt::gtensor_device<gt::blas::index_t, 2> d_perm(gt::shape(n, nbatch));
  gt::gtensor_device<CT, 3> d_Acalc(h_Aactual.shape());
  gt::gtensor<CT, 3> h_Acalc(h_Aactual.shape());
  lumm_batched(d_Acalc, d_LU, d_piv, d_perm, use_pivot);
  gt::copy(d_Acalc, h_Acalc);
  gt::synchronize();

  for (int b = 0; b < nbatch; b++) {
    debug_print_matrix("Acalc", h_Acalc.view(_all, _all, b));
  }

  CT* Acalc_p = gt::raw_pointer_cast(h_Acalc.data());
  CT* Aactual_p = gt::raw_pointer_cast(h_Aactual.data());

  T max_err = T(0);
  T err;
  for (int i = 0; i < n * n * nbatch; i++) {
    err = gt::norm(Acalc_p[i] - Aactual_p[i]);
    if (err > max_err) {
      max_err = err;
    }
  }

  if (max_err > tol) {
    std::cerr << "ERR: max absolute error " << max_err << " > " << tol
              << std::endl;
    return false;
  }
  return true;
}

// ======================================================================
// BM_getrf
//

template <typename R, int N, int NBATCH, bool PIVOT>
static void BM_getrf(benchmark::State& state)
{
  using CT = gt::complex<R>;

  auto h_A = make_test_matrix<CT>(N, N - 1, NBATCH, PIVOT);
  auto h_Acopy = gt::empty_like(h_A);
  gt::gtensor_device<CT, 3> d_A(h_A.shape());

  gt::gtensor<CT*, 1> h_Aptr(NBATCH);
  gt::gtensor_device<CT*, 1> d_Aptr(NBATCH);

  gt::gtensor<gt::blas::index_t, 2> h_piv(gt::shape(N, NBATCH));
  gt::gtensor_device<gt::blas::index_t, 2> d_piv(gt::shape(N, NBATCH));
  gt::gtensor_device<int, 1> d_info(NBATCH);

  h_Aptr(0) = gt::raw_pointer_cast(d_A.data());
  for (int b = 1; b < NBATCH; b++) {
    h_Aptr(b) = h_Aptr(0) + (N * N * b);
  }

  h_Acopy = h_A;
  gt::copy(h_A, d_A);
  gt::copy(h_Aptr, d_Aptr);
  gt::synchronize();

  gt::blas::handle_t* h = gt::blas::create();

  auto fn = [&]() {
    if (PIVOT) {
      gt::blas::getrf_batched(h, N, gt::raw_pointer_cast(d_Aptr.data()), N,
                              gt::raw_pointer_cast(d_piv.data()),
                              gt::raw_pointer_cast(d_info.data()), NBATCH);
    } else {
      gt::blas::getrf_npvt_batched(h, N, gt::raw_pointer_cast(d_Aptr.data()), N,
                                   gt::raw_pointer_cast(d_info.data()), NBATCH);
    }
    gt::synchronize();
  };

  // warm up, device compile, check
  fn();
  check_lu_batched<R>(h_Acopy, d_A, d_piv, PIVOT);
  gt::copy(d_A, h_A);
  gt::copy(d_piv, h_piv);

  for (int b = 0; b < NBATCH; b++) {
    debug_print_matrix("A", h_Acopy.view(_all, _all, b));
    debug_print_matrix("Alu", h_A.view(_all, _all, b));
    debug_print_vector("piv", h_piv.view(_all, b));
  }

  for (auto _ : state) {
    fn();
  }

  gt::blas::destroy(h);
}

// RealType, N, NBATCH, Pivot
BENCHMARK(BM_getrf<float, 512, 128, true>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_getrf<double, 512, 128, true>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_getrf<float, 512, 128, false>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_getrf<double, 512, 128, false>)->Unit(benchmark::kMillisecond);

BENCHMARK(BM_getrf<float, 210, 512, true>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_getrf<double, 210, 512, true>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_getrf<float, 210, 512, false>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_getrf<double, 210, 512, false>)->Unit(benchmark::kMillisecond);

// small cases for debugging
/*
BENCHMARK(BM_getrf<double, 5, 2, false>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_getrf<double, 5, 2, true>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_getrf<float, 5, 2, false>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_getrf<float, 5, 2, true>)->Unit(benchmark::kMillisecond);
*/

BENCHMARK_MAIN();
