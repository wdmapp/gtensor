#include <cmath>
#include <iostream>

#include <benchmark/benchmark.h>

#include <gtensor/gtensor.h>

#include "gt-bm.h"

using namespace gt::placeholders;

constexpr int KB = 1024;
constexpr int MB = 1024 * KB;

static const gt::gtensor<double, 1> stencil_arrays[] = {
  {},
  {},
  {},
  {-0.5, 0.0, 0.5},
  {},
  {1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0},
  {},
  {-1.0 / 60.0, 3.0 / 20.0, -3.0 / 4.0, 0.0, 3.0 / 4.0, -3.0 / 20.0,
   1.0 / 60.0},
  {},
  {1.0 / 280.0, -4.0 / 105.0, 1.0 / 5.0, -4.0 / 5.0, 0.0, 4.0 / 5.0, -1.0 / 5.0,
   4.0 / 105.0, -1.0 / 280.0},
  {},
  {-1. / 1260, 5. / 504, -5. / 84, 5. / 21, -5. / 6, 0., 5. / 6, -5. / 21,
   5. / 84, -5. / 504, 1. / 1260},
  {},
  {1. / 5544, -1. / 385, 1. / 56, -5. / 63, 15. / 56, -6. / 7, 0., 6. / 7,
   -15. / 56, 5. / 63, -1. / 56, 1. / 385, -1. / 5544},
};

namespace detail
{

template <typename S, int N, int i>
struct stencil_builder
{
  static auto expr(const gt::bm::gtensor2<double, 1, S>& y,
                   const gt::gtensor<double, 1> stencil)
  {
    const int bnd = N - 1;
    return stencil(i) * y.view(_s(i, i - bnd)) +
           stencil_builder<S, N, i - 1>::expr(y, stencil);
  }
};

template <typename S, int N>
struct stencil_builder<S, N, 0>
{
  static auto expr(const gt::bm::gtensor2<double, 1, S>& y,
                   const gt::gtensor<double, 1> stencil)
  {
    const int bnd = N - 1;
    return stencil(0) * y.view(_s(0, -bnd));
  }
};

} // namespace detail

template <typename S, int N>
static auto stencil_expr(const gt::bm::gtensor2<double, 1, S>& y)
{
  return detail::stencil_builder<S, N, N - 1>::expr(y, stencil_arrays[N]);
}

// Trick for compile type type debugging, by forcing printing the type in an
// error message. Uncomment code below to use. Useful for verifying that an
// expression remains unevaluated.
// template <typename...> struct WhichType;

template <typename S, int N>
static void BM_stencil1d(benchmark::State& state)
{
  int n = state.range(0) * MB;

  gt::gtensor<double, 1> x(gt::shape(n));
  gt::gtensor<double, 1> y = gt::empty_like(x);
  gt::bm::gtensor2<double, 1, S> d_y(gt::shape(n));
  gt::gtensor<double, 1> dydx_numeric(gt::shape(n));
  gt::bm::gtensor2<double, 1, S> d_dydx_numeric(gt::shape(n));
  double lx = 8;
  double dx = n / lx;
  double scale = lx / n;
  auto fn = [](double x) { return x * x * x; };
  auto fn_dydx = [](double x) { return 3.0 * x * x; };

  for (int i = 0; i < x.shape(0); i++) {
    double xtmp = i * dx;
    x(i) = xtmp;
    y(i) = fn(xtmp);
  }

  gt::copy(y, d_y);
  gt::synchronize();

  // expression object, unevaluated until assigning to a container
  auto&& expr = stencil_expr<S, N>(d_y) * scale;

  // Compile time type check by error message, uncomment WhichType above to use
  // auto&& k_expr = expr.to_kernel();
  // WhichType<decltype(k_expr)> expr_type;

  // warm up, force compile
  d_dydx_numeric = expr;
  gt::synchronize();

  for (auto _ : state) {
    d_dydx_numeric = expr;
    gt::synchronize();
  }
}

BENCHMARK(BM_stencil1d<gt::space::device, 3>)
  ->Range(8, 256)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_stencil1d<gt::space::device, 5>)
  ->Range(8, 256)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_stencil1d<gt::space::device, 7>)
  ->Range(8, 256)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_stencil1d<gt::space::device, 9>)
  ->Range(8, 256)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_stencil1d<gt::space::device, 11>)
  ->Range(8, 256)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_stencil1d<gt::space::device, 13>)
  ->Range(8, 256)
  ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_stencil1d<gt::space::managed, 3>)
  ->Range(8, 256)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_stencil1d<gt::space::managed, 5>)
  ->Range(8, 256)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_stencil1d<gt::space::managed, 7>)
  ->Range(8, 256)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_stencil1d<gt::space::managed, 9>)
  ->Range(8, 256)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_stencil1d<gt::space::managed, 11>)
  ->Range(8, 256)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_stencil1d<gt::space::managed, 13>)
  ->Range(8, 256)
  ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
