
#include <benchmark/benchmark.h>

#include <gtensor/gtensor.h>

using namespace gt::placeholders;

using real_t = double;
using complex_t = gt::complex<double>;

#ifndef GTENSOR_BENCHMARK_PER_DIM_SIZE
#define GTENSOR_BENCHMARK_PER_DIM_SIZE 128
#endif

// ======================================================================
// BM_gt_device_memcpy

template <typename T>
static void BM_gt_device_memcpy(benchmark::State& state)
{
  int n = state.range(0);
  int N = n * n * n * n;

  auto a = gt::empty_device<T>({N});
  auto b = gt::empty_device<T>({N});

  for (auto _ : state) {
    gt::copy_n(a.data(), a.size(), b.data());
    gt::synchronize();
  }
}

BENCHMARK(BM_gt_device_memcpy<double>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_gt_device_memcpy<float>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);

// ======================================================================
// BM_gt_device_copy

template <typename T>
static void BM_gt_device_copy(benchmark::State& state)
{
  int n = state.range(0);
  int N = n * n * n * n;

  auto a = gt::empty_device<T>({N});
  auto b = gt::empty_device<T>({N});

  for (auto _ : state) {
    gt::copy(a, b);
    gt::synchronize();
  }
}

BENCHMARK(BM_gt_device_copy<double>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_gt_device_copy<float>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);

// ======================================================================
// BM_device_assign_1d

template <typename T>
static void BM_device_assign_1d(benchmark::State& state)
{
  int n = state.range(0);
  int N = n * n * n * n;
  auto a = gt::zeros_device<T>(gt::shape(N));
  auto b = gt::empty_like(a);

  // warmup, device compile
  b = a;
  gt::synchronize();

  for (auto _ : state) {
    b = a;
    gt::synchronize();
  }
}

BENCHMARK(BM_device_assign_1d<double>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign_1d<float>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);

// ======================================================================
// BM_device_assign_2d

template <typename T>
static void BM_device_assign_2d(benchmark::State& state)
{
  int n = state.range(0);
  int N = n * n;
  auto a = gt::zeros_device<T>(gt::shape(N, N));
  auto b = gt::empty_like(a);

  // warmup, device compile
  b = a;
  gt::synchronize();

  for (auto _ : state) {
    b = a;
    gt::synchronize();
  }
}

BENCHMARK(BM_device_assign_2d<double>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign_2d<float>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);

// ======================================================================
// BM_device_assign_3d

template <typename T>
static void BM_device_assign_3d(benchmark::State& state)
{
  int n = state.range(0);
  auto a = gt::zeros_device<T>(gt::shape(n * n, n, n));
  auto b = gt::empty_like(a);

  // warmup, device compile
  b = a;
  gt::synchronize();

  for (auto _ : state) {
    b = a;
    gt::synchronize();
  }
}

BENCHMARK(BM_device_assign_3d<double>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign_3d<float>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);

// ======================================================================
// BM_device_assign_4d

template <typename T>
static void BM_device_assign_4d(benchmark::State& state)
{
  int n = state.range(0);
  auto a = gt::zeros_device<T>(gt::shape(n, n, n, n));
  auto b = gt::empty_like(a);

  // warmup, device compile
  b = a;
  gt::synchronize();

  for (auto _ : state) {
    b = a;
    gt::synchronize();
  }
}

BENCHMARK(BM_device_assign_4d<double>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign_4d<float>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);

// ======================================================================
// BM_device_assign_4d_launch

template <typename T>
static void BM_device_assign_4d_launch(benchmark::State& state)
{
  int n = state.range(0);
  auto a = gt::zeros_device<T>(gt::shape(n, n, n, n));
  auto b = gt::empty_like(a);

  auto k_a = a.to_kernel();
  auto k_b = b.to_kernel();

  auto assign_lambda = GT_LAMBDA(int i0, int i1, int i2, int i3)
  {
    k_b(i0, i1, i2, i3) = k_a(i0, i1, i2, i3);
  };

  // warmup, device compile
  gt::launch<4>(a.shape(), assign_lambda);
  gt::synchronize();

  for (auto _ : state) {
    gt::launch<4>(a.shape(), assign_lambda);
    gt::synchronize();
  }
}

BENCHMARK(BM_device_assign_4d_launch<double>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign_4d_launch<float>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);

// ======================================================================
// BM_device_assign_1d_op

template <typename T>
static void BM_device_assign_1d_op(benchmark::State& state)
{
  int n = state.range(0);
  int N = n * n * n * n;
  auto a = gt::zeros_device<T>(gt::shape(N));
  auto b = gt::empty_like(a);

  // warmup, device compile
  auto expr = a + 2 * a;
  b = expr;
  gt::synchronize();

  for (auto _ : state) {
    b = expr;
    gt::synchronize();
  }
}

BENCHMARK(BM_device_assign_1d_op<double>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign_1d_op<float>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);

// ======================================================================
// BM_device_assign_2d_op

template <typename T>
static void BM_device_assign_2d_op(benchmark::State& state)
{
  int n = state.range(0);
  int N = n * n;
  auto a = gt::zeros_device<T>(gt::shape(N, N));
  auto b = gt::empty_like(a);

  // warmup, device compile
  auto expr = a + 2 * a;
  b = expr;
  gt::synchronize();

  for (auto _ : state) {
    b = expr;
    gt::synchronize();
  }
}

BENCHMARK(BM_device_assign_2d_op<double>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign_2d_op<float>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);

// ======================================================================
// BM_device_assign_3d_op

template <typename T>
static void BM_device_assign_3d_op(benchmark::State& state)
{
  int n = state.range(0);
  auto a = gt::zeros_device<T>(gt::shape(n * n, n, n));
  auto b = gt::empty_like(a);

  // warmup, device compile
  auto expr = a + 2 * a;
  b = expr;
  gt::synchronize();

  for (auto _ : state) {
    b = expr;
    gt::synchronize();
  }
}

BENCHMARK(BM_device_assign_3d_op<double>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign_3d_op<float>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);

// ======================================================================
// BM_device_assign_4d_op

template <typename T>
static void BM_device_assign_4d_op(benchmark::State& state)
{
  int n = state.range(0);
  auto a = gt::zeros_device<T>(gt::shape(n, n, n, n));
  auto b = gt::empty_like(a);

  // warmup, device compile
  b = a + 2 * a;
  gt::synchronize();

  for (auto _ : state) {
    b = a + 2 * a;
    gt::synchronize();
  }
}

BENCHMARK(BM_device_assign_4d_op<double>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign_4d_op<float>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);

// ======================================================================
// BM_device_assign_4d_op_launch

template <typename T>
static void BM_device_assign_4d_op_launch(benchmark::State& state)
{
  int n = state.range(0);
  auto a = gt::zeros_device<T>(gt::shape(n, n, n, n));
  auto b = gt::empty_like(a);

  auto k_a = a.to_kernel();
  auto k_b = b.to_kernel();

  auto op_lambda = GT_LAMBDA(int i0, int i1, int i2, int i3)
  {
    k_b(i0, i1, i2, i3) = k_a(i0, i1, i2, i3) + 2 * k_a(i0, i1, i2, i3);
  };

  // warmup, device compile
  gt::launch<4>(a.shape(), op_lambda);
  gt::synchronize();

  for (auto _ : state) {
    gt::launch<4>(a.shape(), op_lambda);
    gt::synchronize();
  }
}

BENCHMARK(BM_device_assign_4d_op_launch<double>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign_4d_op_launch<float>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
