
#include <benchmark/benchmark.h>

#include <gtensor/gtensor.h>

using namespace gt::placeholders;

using real_t = double;
using complex_t = gt::complex<double>;

#ifndef GTENSOR_BENCHMARK_PER_DIM_SIZE
#define GTENSOR_BENCHMARK_PER_DIM_SIZE 128
#endif

namespace gt
{

namespace bm
{

template <int Dim>
auto get_shape(int n_per_dim)
{
  gt::shape_type<Dim> shape;
  int N = n_per_dim * n_per_dim * n_per_dim * n_per_dim;
  if (Dim == 1) {
    shape[0] = N;
  } else if (Dim == 2) {
    shape[0] = n_per_dim * n_per_dim;
    shape[1] = n_per_dim * n_per_dim;
  } else if (Dim == 3) {
    shape[0] = n_per_dim * n_per_dim;
    shape[1] = n_per_dim;
    shape[2] = n_per_dim;
  } else {
    shape[0] = n_per_dim;
    shape[1] = n_per_dim;
    shape[2] = n_per_dim;
    shape[3] = n_per_dim;
  }
  return shape;
}

} // namespace bm

} // namespace gt

// ======================================================================
// BM_device_memcpy

template <typename T>
static void BM_device_memcpy(benchmark::State& state)
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

BENCHMARK(BM_device_memcpy<double>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_memcpy<float>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);

// ======================================================================
// BM_device_copy

template <typename T>
static void BM_device_copy(benchmark::State& state)
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

BENCHMARK(BM_device_copy<double>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_copy<float>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);

// ======================================================================
// BM_device_assign

template <typename T, int Dim>
static void BM_device_assign(benchmark::State& state)
{
  int n = state.range(0);
  gt::shape_type<Dim> shape = gt::bm::get_shape<Dim>(n);

  auto a = gt::zeros_device<T>(shape);
  auto b = gt::empty_like(a);

  // warmup, device compile
  b = a;
  gt::synchronize();

  for (auto _ : state) {
    b = a;
    gt::synchronize();
  }
}

BENCHMARK(BM_device_assign<double, 1>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign<float, 1>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign<double, 2>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign<float, 2>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign<double, 3>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign<float, 3>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign<double, 4>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign<float, 4>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);

// ======================================================================
// BM_device_assign_launch

template <typename T>
static void BM_device_assign_launch_1d(benchmark::State& state)
{
  int n = state.range(0);
  gt::shape_type<1> shape = gt::bm::get_shape<1>(n);

  auto a = gt::zeros_device<T>(shape);
  auto b = gt::empty_like(a);

  auto k_a = a.to_kernel();
  auto k_b = b.to_kernel();

  auto assign_lambda = GT_LAMBDA(int i0) { k_b(i0) = k_a(i0); };
  // warmup, device compile
  gt::launch<1>(a.shape(), assign_lambda);
  gt::synchronize();

  for (auto _ : state) {
    gt::launch<1>(a.shape(), assign_lambda);
    gt::synchronize();
  }
}

template <typename T>
static void BM_device_assign_launch_2d(benchmark::State& state)
{
  int n = state.range(0);
  gt::shape_type<2> shape = gt::bm::get_shape<2>(n);

  auto a = gt::zeros_device<T>(shape);
  auto b = gt::empty_like(a);

  auto k_a = a.to_kernel();
  auto k_b = b.to_kernel();

  auto assign_lambda = GT_LAMBDA(int i0, int i1) { k_b(i0, i1) = k_a(i0, i1); };
  // warmup, device compile
  gt::launch<2>(a.shape(), assign_lambda);
  gt::synchronize();

  for (auto _ : state) {
    gt::launch<2>(a.shape(), assign_lambda);
    gt::synchronize();
  }
}

template <typename T>
static void BM_device_assign_launch_3d(benchmark::State& state)
{
  int n = state.range(0);
  gt::shape_type<3> shape = gt::bm::get_shape<3>(n);

  auto a = gt::zeros_device<T>(shape);
  auto b = gt::empty_like(a);

  auto k_a = a.to_kernel();
  auto k_b = b.to_kernel();

  auto assign_lambda = GT_LAMBDA(int i0, int i1, int i2)
  {
    k_b(i0, i1, i2) = k_a(i0, i1, i2);
  };

  // warmup, device compile
  gt::launch<3>(a.shape(), assign_lambda);
  gt::synchronize();

  for (auto _ : state) {
    gt::launch<3>(a.shape(), assign_lambda);
    gt::synchronize();
  }
}

template <typename T>
static void BM_device_assign_launch_4d(benchmark::State& state)
{
  int n = state.range(0);
  gt::shape_type<4> shape = gt::bm::get_shape<4>(n);

  auto a = gt::zeros_device<T>(shape);
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

BENCHMARK(BM_device_assign_launch_1d<double>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign_launch_1d<float>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_device_assign_launch_2d<double>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign_launch_2d<float>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_device_assign_launch_3d<double>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign_launch_3d<float>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_device_assign_launch_4d<double>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign_launch_4d<float>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);

// ======================================================================
// BM_device_assign_op

template <typename T, int Dim>
static void BM_device_assign_op(benchmark::State& state)
{
  int n = state.range(0);
  gt::shape_type<Dim> shape = gt::bm::get_shape<Dim>(n);

  auto a = gt::zeros_device<T>(shape);
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

BENCHMARK(BM_device_assign_op<double, 1>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign_op<float, 1>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign_op<double, 2>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign_op<float, 2>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign_op<double, 3>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign_op<float, 3>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign_op<double, 4>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign_op<float, 4>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);

// ======================================================================
// BM_device_assign_[1-4]d_op_launch

template <typename T>
static void BM_device_assign_op_launch_1d(benchmark::State& state)
{
  int n = state.range(0);
  gt::shape_type<1> shape = gt::bm::get_shape<1>(n);
  auto a = gt::zeros_device<T>(shape);
  auto b = gt::empty_like(a);

  auto k_a = a.to_kernel();
  auto k_b = b.to_kernel();

  auto op_lambda = GT_LAMBDA(int i0) { k_b(i0) = k_a(i0) + 2 * k_a(i0); };

  // warmup, device compile
  gt::launch<1>(a.shape(), op_lambda);
  gt::synchronize();

  for (auto _ : state) {
    gt::launch<1>(a.shape(), op_lambda);
    gt::synchronize();
  }
}

BENCHMARK(BM_device_assign_op_launch_1d<double>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign_op_launch_1d<float>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);

template <typename T>
static void BM_device_assign_op_launch_2d(benchmark::State& state)
{
  int n = state.range(0);
  gt::shape_type<2> shape = gt::bm::get_shape<2>(n);
  auto a = gt::zeros_device<T>(shape);
  auto b = gt::empty_like(a);

  auto k_a = a.to_kernel();
  auto k_b = b.to_kernel();

  auto op_lambda = GT_LAMBDA(int i0, int i1)
  {
    k_b(i0, i1) = k_a(i0, i1) + 2 * k_a(i0, i1);
  };

  // warmup, device compile
  gt::launch<2>(a.shape(), op_lambda);
  gt::synchronize();

  for (auto _ : state) {
    gt::launch<2>(a.shape(), op_lambda);
    gt::synchronize();
  }
}

BENCHMARK(BM_device_assign_op_launch_2d<double>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign_op_launch_2d<float>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);

template <typename T>
static void BM_device_assign_op_launch_3d(benchmark::State& state)
{
  int n = state.range(0);
  gt::shape_type<3> shape = gt::bm::get_shape<3>(n);
  auto a = gt::zeros_device<T>(shape);
  auto b = gt::empty_like(a);

  auto k_a = a.to_kernel();
  auto k_b = b.to_kernel();

  auto op_lambda = GT_LAMBDA(int i0, int i1, int i2)
  {
    k_b(i0, i1, i2) = k_a(i0, i1, i2) + 2 * k_a(i0, i1, i2);
  };

  // warmup, device compile
  gt::launch<3>(a.shape(), op_lambda);
  gt::synchronize();

  for (auto _ : state) {
    gt::launch<3>(a.shape(), op_lambda);
    gt::synchronize();
  }
}

BENCHMARK(BM_device_assign_op_launch_3d<double>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign_op_launch_3d<float>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);

template <typename T>
static void BM_device_assign_op_launch_4d(benchmark::State& state)
{
  int n = state.range(0);
  gt::shape_type<4> shape = gt::bm::get_shape<4>(n);
  auto a = gt::zeros_device<T>(shape);
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

BENCHMARK(BM_device_assign_op_launch_4d<double>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign_op_launch_4d<float>)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE - 1)
  ->Arg(GTENSOR_BENCHMARK_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
