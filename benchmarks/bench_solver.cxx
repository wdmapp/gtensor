#include <iostream>
#include <numeric>
#include <string>

#include <benchmark/benchmark.h>

#include <gtensor/gtensor.h>

#include <gt-solver/solver.h>

#include "gt-bm.h"

using namespace gt::placeholders;

// #define BENCH_SOLVER_DEBUG

template <typename T>
auto make_test_matrix(int n, int bw, int batch_size, bool needs_pivot)
{
  auto h_Adata = gt::zeros<T>({n, n, batch_size});
  for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < n; i++) {
      h_Adata(i, i, b) = T(bw + 1.0);
      // set upper / lower diags at bw diagonal
      for (int d = 1; d <= bw; d++) {
        if (i + d < n) {
          h_Adata(i, i + d, b) = T(-1.0);
          h_Adata(i + d, i, b) = T(-0.5);
        }
      }
    }
    if (needs_pivot) {
      h_Adata(0, 0, b) = T(n / 64.0);
    }
  }
  return h_Adata;
}

// ======================================================================
// BM_solver
//

// args: int N, int BW, int NRHS, int NBATCH
template <typename Solver>
static void BM_solver(benchmark::State& state)
{
  using T = typename Solver::value_type;
  using R = gt::complex_subtype_t<T>;

  const int N = state.range(0);
  const int BW = state.range(1);
  const int NRHS = state.range(2);
  const int NBATCH = state.range(3);

  auto h_A = make_test_matrix<T>(N, BW, NBATCH, true);

  gt::gtensor<T, 3> h_rhs(gt::shape(N, NRHS, NBATCH));
  gt::gtensor<T, 3> h_result(h_rhs.shape());
  gt::gtensor_device<T, 3> d_rhs(h_rhs.shape());
  gt::gtensor_device<T, 3> d_result(h_rhs.shape());

  gt::gtensor<T*, 1> h_Aptr(NBATCH);

  h_Aptr(0) = gt::raw_pointer_cast(h_A.data());
  for (int b = 1; b < NBATCH; b++) {
    h_Aptr(b) = h_Aptr(0) + (N * N * b);
    for (int i = 0; i < N; i++) {
      for (int rhs = 0; rhs < NRHS; rhs++) {
        h_rhs(i, rhs, b) = T(1.0 + R(rhs) / NRHS);
      }
    }
  }

  gt::copy(h_rhs, d_rhs);

  gt::blas::handle_t h;

  Solver s(h, N, NBATCH, NRHS, gt::raw_pointer_cast(h_Aptr.data()));

  auto fn = [&]() {
    s.solve(d_rhs.data().get(), d_result.data().get());
    gt::synchronize();
  };

  // warm up, device compile, check
  fn();

  for (auto _ : state) {
    fn();
  }
}

// Solver, N, BW, NRHS, NBATCH
BENCHMARK(BM_solver<gt::solver::solver_dense<double>>)
  ->Args({512, 32, 1, 64})
  ->Args({210, 32, 1, 256})
  ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_solver<gt::solver::solver_invert<double>>)
  ->Args({512, 32, 1, 64})
  ->Args({210, 32, 1, 256})
  ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_solver<gt::solver::solver_sparse<double>>)
  ->Args({512, 32, 1, 64})
  ->Args({512, 5, 1, 64})
  ->Args({210, 32, 1, 256})
  ->Args({210, 5, 1, 256})
  ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_solver<gt::solver::solver_dense<gt::complex<double>>>)
  ->Args({512, 32, 1, 64})
  ->Args({210, 32, 1, 256})
  ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_solver<gt::solver::solver_invert<gt::complex<double>>>)
  ->Args({512, 32, 1, 64})
  ->Args({210, 32, 1, 256})
  ->Unit(benchmark::kMillisecond);

// oneMKL sparse API does not yet support complex
#ifndef GTENSOR_DEVICE_SYCL
BENCHMARK(BM_solver<gt::solver::solver_sparse<gt::complex<double>>>)
  ->Args({512, 32, 1, 64})
  ->Args({512, 5, 1, 64})
  ->Args({210, 32, 1, 256})
  ->Args({210, 5, 1, 256})
  ->Unit(benchmark::kMillisecond);
#endif

BENCHMARK_MAIN();
