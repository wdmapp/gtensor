#include <cmath>
#include <iostream>
#include <time.h>

#include <gtensor/gtensor.h>

using namespace gt::placeholders;

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

void bench_stencils(int n, double lx, double (*fn)(double),
                    double (*dydx)(double))
{
  gt::gtensor<double, 1> x(gt::shape(n));
  gt::gtensor<double, 1> y = gt::empty_like(x);
  gt::gtensor_device<double, 1> d_y(gt::shape(n));
  gt::gtensor<double, 1> dydx_analytic = gt::empty_like(x);
  gt::gtensor<double, 1> dydx_numeric(gt::shape(n));
  gt::gtensor_device<double, 1> d_dydx_numeric(gt::shape(n));
  double dx = n / lx;
  double scale = lx / n;

  struct timespec start, end;
  double seconds_per_run = 0.0;

  for (int i = 0; i < x.shape(0); i++) {
    double xtmp = i * dx;
    x(i) = xtmp;
    y(i) = fn(xtmp);
    dydx_analytic(i) = dydx(xtmp);
  }
  /*
  std::cout << "x            = " << x << std::endl;
  std::cout << "y            = " << y << std::endl;
  std::cout << "dydx_analytic = " << dydx_analytic << std::endl;
  std::cout << std::endl;
  */

  gt::copy(y, d_y);

  auto bench = [&](const char* label, auto&& f, int run_count = 8,
                   int warm_up_count = 2) {
    // warmup run
    for (int i = 0; i < warm_up_count; i++) {
      f();
    }
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < run_count; i++) {
      f();
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    seconds_per_run =
      ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1.0e-9) /
      run_count;
    std::cout << label << "\t" << seconds_per_run << std::endl;
  };

  bench("3", [&]() {
    d_dydx_numeric = stencil1d_3<gt::space::device>(d_y, stencil3) * scale;
  });
  bench("5", [&]() {
    d_dydx_numeric = stencil1d_5<gt::space::device>(d_y, stencil5) * scale;
  });
  bench("7", [&]() {
    d_dydx_numeric = stencil1d_7<gt::space::device>(d_y, stencil7) * scale;
  });
}

int main(int argc, char** argv)
{
  int n = 16;
  int lx = 8;
  if (argc > 1) {
    n = std::stoi(argv[1]);
  }

  bench_stencils(
    n, lx, [](double x) { return x * x * x; },
    [](double x) { return 3.0 * x * x; });
}
