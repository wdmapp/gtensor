#include <cmath>
#include <iostream>

#include <benchmark/benchmark.h>

#include <gtensor/gtensor.h>

#include "gt-bm.h"

using namespace gt::placeholders;

constexpr int KB = 1024;
constexpr int MB = 1024 * KB;

static const gt::gtensor<double, 1> stencil3 = {-0.5, 0.0, 0.5};
static const gt::gtensor<double, 1> stencil5 = {1.0 / 12.0, -2.0 / 3.0, 0.0,
                                                2.0 / 3.0, -1.0 / 12.0};
static const gt::gtensor<double, 1> stencil7 = {
  -1.0 / 60.0, 3.0 / 20.0, -3.0 / 4.0, 0.0, 3.0 / 4.0, -3.0 / 20.0, 1.0 / 60.0};

static const gt::gtensor<double, 1> stencil9 = {
  1.0 / 280.0, -4.0 / 105.0, 1.0 / 5.0,   -4.0 / 5.0,  0.0,
  4.0 / 5.0,   -1.0 / 5.0,   4.0 / 105.0, -1.0 / 280.0};

static const gt::gtensor<double, 1> stencil11 = {
  -1. / 1260, 5. / 504, -5. / 84, 5. / 21,   -5. / 6,  0.,
  5. / 6,     -5. / 21, 5. / 84,  -5. / 504, 1. / 1260};

static const gt::gtensor<double, 1> stencil13 = {
  1. / 5544, -1. / 385, 1. / 56, -5. / 63, 15. / 56, -6. / 7,   0.,
  6. / 7,    -15. / 56, 5. / 63, -1. / 56, 1. / 385, -1. / 5544};

template <typename S>
inline auto stencil1d_3(const gt::bm::gtensor2<double, 1, S>& y,
                        const gt::gtensor<double, 1>& stencil)
{
  return stencil(0) * y.view(_s(0, -2)) + stencil(1) * y.view(_s(1, -1)) +
         stencil(2) * y.view(_s(2, _));
}

template <typename S>
inline auto stencil1d_5(const gt::bm::gtensor2<double, 1, S>& y,
                        const gt::gtensor<double, 1>& stencil)
{
  return stencil(0) * y.view(_s(0, -4)) + stencil(1) * y.view(_s(1, -3)) +
         stencil(2) * y.view(_s(2, -2)) + stencil(3) * y.view(_s(3, -1)) +
         stencil(4) * y.view(_s(4, _));
}

template <typename S>
inline auto stencil1d_7(const gt::bm::gtensor2<double, 1, S>& y,
                        const gt::gtensor<double, 1>& stencil)
{
  return stencil(0) * y.view(_s(0, -6)) + stencil(1) * y.view(_s(1, -5)) +
         stencil(2) * y.view(_s(2, -4)) + stencil(3) * y.view(_s(3, -3)) +
         stencil(4) * y.view(_s(4, -2)) + stencil(5) * y.view(_s(5, -1)) +
         stencil(6) * y.view(_s(6, _));
}

template <typename S>
inline auto stencil1d_9(const gt::bm::gtensor2<double, 1, S>& y,
                        const gt::gtensor<double, 1>& stencil)
{
  return stencil(0) * y.view(_s(0, -8)) + stencil(1) * y.view(_s(1, -7)) +
         stencil(2) * y.view(_s(2, -6)) + stencil(3) * y.view(_s(3, -5)) +
         stencil(4) * y.view(_s(4, -4)) + stencil(5) * y.view(_s(5, -3)) +
         stencil(6) * y.view(_s(6, -2)) + stencil(7) * y.view(_s(7, -1)) +
         stencil(8) * y.view(_s(8, _));
}

template <typename S>
inline auto stencil1d_11(const gt::bm::gtensor2<double, 1, S>& y,
                         const gt::gtensor<double, 1>& stencil)
{
  return stencil(0) * y.view(_s(0, -10)) + stencil(1) * y.view(_s(1, -9)) +
         stencil(2) * y.view(_s(2, -8)) + stencil(3) * y.view(_s(3, -7)) +
         stencil(4) * y.view(_s(4, -6)) + stencil(5) * y.view(_s(5, -5)) +
         stencil(6) * y.view(_s(6, -4)) + stencil(7) * y.view(_s(7, -3)) +
         stencil(8) * y.view(_s(8, -2)) + stencil(9) * y.view(_s(9, -1)) +
         stencil(10) * y.view(_s(10, _));
}

template <typename S>
inline auto stencil1d_13(const gt::bm::gtensor2<double, 1, S>& y,
                         const gt::gtensor<double, 1>& stencil)
{
  return stencil(0) * y.view(_s(0, -12)) + stencil(1) * y.view(_s(1, -11)) +
         stencil(2) * y.view(_s(2, -10)) + stencil(3) * y.view(_s(3, -9)) +
         stencil(4) * y.view(_s(4, -8)) + stencil(5) * y.view(_s(5, -7)) +
         stencil(6) * y.view(_s(6, -6)) + stencil(7) * y.view(_s(7, -5)) +
         stencil(8) * y.view(_s(8, -4)) + stencil(9) * y.view(_s(9, -3)) +
         stencil(10) * y.view(_s(10, -2)) + stencil(11) * y.view(_s(11, -1)) +
         stencil(12) * y.view(_s(12, _));
}

// Template meta programming trick to allow a templated benchmark below. Note
// that the expression type returned by each stencil routines is different, so
// using a single routine for all of them is not possible. This also prevents
// building an expression up iteratively into a variable, since the type
// changes each time.
namespace detail
{

template <int N>
struct stencil;

template <>
struct stencil<3>
{
  template <typename S>
  static auto compute(const gt::bm::gtensor2<double, 1, S>& y)
  {
    return stencil1d_3<S>(y, stencil3);
  }
};

template <>
struct stencil<5>
{
  template <typename S>
  static auto compute(const gt::bm::gtensor2<double, 1, S>& y)
  {
    return stencil1d_5<S>(y, stencil5);
  }
};

template <>
struct stencil<7>
{
  template <typename S>
  static auto compute(const gt::bm::gtensor2<double, 1, S>& y)
  {
    return stencil1d_7<S>(y, stencil7);
  }
};

template <>
struct stencil<9>
{
  template <typename S>
  static auto compute(const gt::bm::gtensor2<double, 1, S>& y)
  {
    return stencil1d_9<S>(y, stencil9);
  }
};

template <>
struct stencil<11>
{
  template <typename S>
  static auto compute(const gt::bm::gtensor2<double, 1, S>& y)
  {
    return stencil1d_11<S>(y, stencil11);
  }
};

template <>
struct stencil<13>
{
  template <typename S>
  static auto compute(const gt::bm::gtensor2<double, 1, S>& y)
  {
    return stencil1d_13<S>(y, stencil13);
  }
};

} // namespace detail

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
  auto&& stencil_expr = detail::stencil<N>::template compute<S>(d_y) * scale;

  // Trick to print type in an error at compile time for debugging
  // auto&& k_stencil_expr = stencil_expr.to_kernel();
  // WhichType<decltype(k_stencil_expr)> stencil_expr_type;

  // warm up, force compile
  d_dydx_numeric = stencil_expr;
  gt::synchronize();

  for (auto _ : state) {
    d_dydx_numeric = stencil_expr;
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
