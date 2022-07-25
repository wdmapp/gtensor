#include <iostream>
#include <numeric>
#include <string>

#include <benchmark/benchmark.h>

#include <gtensor/gtensor.h>

#include <gt-blas/blas.h>

#include "gt-bm.h"

using namespace gt::placeholders;

// #define BENCH_GETRI_DEBUG

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

template <typename T, typename CT = gt::complex<T>>
bool check_close_identity_batch(gt::gtensor<CT, 3>& h_A)
{
  int n = h_A.shape(0);
  int nbatch = h_A.shape(2);
  constexpr T tol = 100.0 * std::numeric_limits<T>::epsilon();

  // should be identity matrix in every batch
  T max_err = T(0);
  T err;
  for (int b = 0; b < nbatch; b++) {
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < n; i++) {
        if (i == j) {
          err = gt::norm(h_A(i, j, b) - CT(1.0));
        } else {
          err = gt::norm(h_A(i, j, b));
        }
        if (err > max_err) {
          max_err = err;
        }
      }
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
// BM_getri
//

template <typename R, int N, int NBATCH>
static void BM_getri(benchmark::State& state)
{
  using CT = gt::complex<R>;

  auto h_A = make_test_matrix<CT>(N, N - 1, NBATCH, true);
  auto h_Acopy = gt::empty_like(h_A);
  gt::gtensor_device<CT, 3> d_A(h_A.shape());
  gt::gtensor_device<CT, 3> d_Acopy(h_A.shape());

  gt::gtensor<CT, 3> h_AAinv(h_A.shape());
  gt::gtensor_device<CT, 3> d_AAinv(h_A.shape());

  auto h_C = gt::empty_like(h_A);
  gt::gtensor_device<CT, 3> d_C(h_A.shape());

  gt::gtensor<CT*, 1> h_Aptr(NBATCH);
  gt::gtensor_device<CT*, 1> d_Aptr(NBATCH);

  gt::gtensor<CT*, 1> h_Cptr(NBATCH);
  gt::gtensor_device<CT*, 1> d_Cptr(NBATCH);

  gt::gtensor<CT*, 1> h_AAinvptr(NBATCH);
  gt::gtensor_device<CT*, 1> d_AAinvptr(NBATCH);

  gt::gtensor<gt::blas::index_t, 2> h_piv(gt::shape(N, NBATCH));
  gt::gtensor_device<gt::blas::index_t, 2> d_piv(gt::shape(N, NBATCH));
  gt::gtensor_device<int, 1> d_info(NBATCH);

  h_Aptr(0) = gt::raw_pointer_cast(d_A.data());
  h_Cptr(0) = gt::raw_pointer_cast(d_C.data());
  h_AAinvptr(0) = gt::raw_pointer_cast(d_AAinv.data());
  for (int b = 1; b < NBATCH; b++) {
    h_Aptr(b) = h_Aptr(0) + (N * N * b);
    h_Cptr(b) = h_Cptr(0) + (N * N * b);
    h_AAinvptr(b) = h_AAinvptr(0) + (N * N * b);
  }

  h_Acopy = h_A;
  gt::copy(h_A, d_A);
  gt::copy(d_A, d_Acopy);
  gt::copy(h_Aptr, d_Aptr);
  gt::copy(h_Cptr, d_Cptr);
  gt::copy(h_AAinvptr, d_AAinvptr);
  gt::synchronize();

  gt::blas::handle_t* h = gt::blas::create();

  gt::blas::getrf_batched(h, N, gt::raw_pointer_cast(d_Aptr.data()), N,
                          gt::raw_pointer_cast(d_piv.data()),
                          gt::raw_pointer_cast(d_info.data()), NBATCH);

  auto fn = [&]() {
    gt::blas::getri_batched(h, N, gt::raw_pointer_cast(d_Aptr.data()), N,
                            gt::raw_pointer_cast(d_piv.data()),
                            gt::raw_pointer_cast(d_Cptr.data()), N,
                            gt::raw_pointer_cast(d_info.data()), NBATCH);
    gt::synchronize();
  };

  // warm up, device compile, check
  fn();

#ifdef BENCH_GETRI_DEBUG
  gt::copy(d_A, h_A);     // now contains Alu
  gt::copy(h_Acopy, d_A); // copy original A back to d_A
  CT a = CT(1.0);
  CT b = CT(0.0);
  gt::blas::gemm_batched(h, N, N, N, a, gt::raw_pointer_cast(d_Aptr.data()), N,
                         gt::raw_pointer_cast(d_Cptr.data()), N, b,
                         gt::raw_pointer_cast(d_AAinvptr.data()), N, NBATCH);
  gt::copy(d_AAinv, h_AAinv);
  check_close_identity_batch<R>(h_AAinv);

  for (int b = 0; b < NBATCH; b++) {
    std::cout << b << " A*Ainv\n" << h_AAinv.view(_all, _all, b) << std::endl;
  }
#endif

  for (auto _ : state) {
    fn();
  }

  gt::blas::destroy(h);
}

// RealType, N, NBATC
BENCHMARK(BM_getri<float, 512, 64>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_getri<double, 512, 64>)->Unit(benchmark::kMillisecond);

BENCHMARK(BM_getri<float, 210, 256>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_getri<double, 210, 256>)->Unit(benchmark::kMillisecond);

// small cases for debugging
/*
BENCHMARK(BM_getri<double, 5, 2>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_getri<float, 5, 2>)->Unit(benchmark::kMillisecond);
*/

BENCHMARK_MAIN();
