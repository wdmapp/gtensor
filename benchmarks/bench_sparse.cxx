#include <iostream>
#include <numeric>
#include <string>

#include <benchmark/benchmark.h>

#include <gtensor/gtensor.h>

#include <gtensor/sparse.h>

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

// ======================================================================
// BM_csr_matrix_from_dense
//

template <typename R, int N, int BW, int NBATCH>
static void BM_csr_matrix_from_dense(benchmark::State& state)
{
  using CT = gt::complex<R>;

  auto h_A = make_test_matrix<CT>(N, BW, NBATCH, true);
  gt::gtensor_device<CT, 3> d_A(h_A.shape());

  gt::copy(h_A, d_A);

  auto fn = [&]() {
    auto d_Acsr =
      gt::sparse::csr_matrix<CT, gt::space::device>::join_matrix_batches(d_A);
    gt::synchronize();
  };

  // warm up, device compile, check
  fn();

  for (auto _ : state) {
    fn();
  }
}

// RealType, N, BW, NBATCH
BENCHMARK(BM_csr_matrix_from_dense<float, 4096, 32, 1>)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_csr_matrix_from_dense<double, 4096, 32, 1>)
  ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_csr_matrix_from_dense<float, 512, 32, 64>)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_csr_matrix_from_dense<double, 512, 32, 64>)
  ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_csr_matrix_from_dense<float, 210, 10, 256>)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_csr_matrix_from_dense<double, 210, 10, 256>)
  ->Unit(benchmark::kMillisecond);

// small cases for debugging
BENCHMARK(BM_csr_matrix_from_dense<double, 8, 2, 2>)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_csr_matrix_from_dense<float, 8, 2, 2>)
  ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
