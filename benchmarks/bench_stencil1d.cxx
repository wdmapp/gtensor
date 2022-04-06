#include <cmath>
#include <iostream>

#include <benchmark/benchmark.h>

#include <gtensor/gtensor.h>

using namespace gt::placeholders;

constexpr int KB = 1024;
constexpr int MB = 1024 * KB;
constexpr int GB = 1024 * MB;

static const gt::gtensor<double, 1> stencil3 = {-0.5, 0.0, 0.5};
static const gt::gtensor<double, 1> stencil5 = {1.0 / 12.0, -2.0 / 3.0, 0.0,
                                                2.0 / 3.0, -1.0 / 12.0};
static const gt::gtensor<double, 1> stencil7 = {
  -1.0 / 60.0, 3.0 / 20.0, -3.0 / 4.0, 0.0, 3.0 / 4.0, -3.0 / 20.0, 1.0 / 60.0};

template <typename S>
inline auto stencil1d_3(const gt::gtensor<double, 1, S>& y,
                        const gt::gtensor<double, 1>& stencil)
{
  return stencil(0) * y.view(_s(0, -2)) + stencil(1) * y.view(_s(1, -1)) +
         stencil(2) * y.view(_s(2, _));
}

template <typename S>
inline auto stencil1d_5(const gt::gtensor<double, 1, S>& y,
                        const gt::gtensor<double, 1>& stencil)
{
  return stencil(0) * y.view(_s(0, -4)) + stencil(1) * y.view(_s(1, -3)) +
         stencil(2) * y.view(_s(2, -2)) + stencil(3) * y.view(_s(3, -1)) +
         stencil(4) * y.view(_s(4, _));
}

template <typename S>
inline auto stencil1d_7(const gt::gtensor<double, 1, S>& y,
                        const gt::gtensor<double, 1>& stencil)
{
  return stencil(0) * y.view(_s(0, -6)) + stencil(1) * y.view(_s(1, -5)) +
         stencil(2) * y.view(_s(2, -4)) + stencil(3) * y.view(_s(3, -3)) +
         stencil(4) * y.view(_s(4, -2)) + stencil(5) * y.view(_s(5, -1)) +
         stencil(6) * y.view(_s(6, _));
}

static void BM_stencil1d_3(benchmark::State& state)
{
  int n = state.range(0) * MB;

  gt::gtensor<double, 1> x(gt::shape(n));
  gt::gtensor<double, 1> y = gt::empty_like(x);
  gt::gtensor_device<double, 1> d_y(gt::shape(n));
  gt::gtensor<double, 1> dydx_numeric(gt::shape(n));
  gt::gtensor_device<double, 1> d_dydx_numeric(gt::shape(n));
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

  for (auto _ : state) {
    d_dydx_numeric = stencil1d_3<gt::space::device>(d_y, stencil3) * scale;
    gt::synchronize();
  }
}

static void BM_stencil1d_5(benchmark::State& state)
{
  int n = state.range(0) * MB;

  gt::gtensor<double, 1> x(gt::shape(n));
  gt::gtensor<double, 1> y = gt::empty_like(x);
  gt::gtensor_device<double, 1> d_y(gt::shape(n));
  gt::gtensor<double, 1> dydx_numeric(gt::shape(n));
  gt::gtensor_device<double, 1> d_dydx_numeric(gt::shape(n));
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

  for (auto _ : state) {
    d_dydx_numeric = stencil1d_5<gt::space::device>(d_y, stencil5) * scale;
    gt::synchronize();
  }
}

static void BM_stencil1d_7(benchmark::State& state)
{
  int n = state.range(0) * MB;

  gt::gtensor<double, 1> x(gt::shape(n));
  gt::gtensor<double, 1> y = gt::empty_like(x);
  gt::gtensor_device<double, 1> d_y(gt::shape(n));
  gt::gtensor<double, 1> dydx_numeric(gt::shape(n));
  gt::gtensor_device<double, 1> d_dydx_numeric(gt::shape(n));
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

  for (auto _ : state) {
    d_dydx_numeric = stencil1d_7<gt::space::device>(d_y, stencil7) * scale;
    gt::synchronize();
  }
}

BENCHMARK(BM_stencil1d_3)->Range(8, 256)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_stencil1d_5)->Range(8, 256)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_stencil1d_7)->Range(8, 256)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
