/*
 * =====================================================================================
 *
 *       Filename:  stencil1d.cxx
 *
 *    Description: Simple 1D stencil example for Gtensor
 *
 *        Version:  1.0
 *        Created:  01/06/2020 06:56:46 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#include <cmath>
#include <iostream>

#include <gtensor/gtensor.h>

// provides convenient shortcuts for commont gtensor functions, for example
// _s for gt::gslice
using namespace gt::placeholders;

static const gt::gtensor<double, 1> stencil3 = {-0.5, 0.0, 0.5};
static const gt::gtensor<double, 1> stencil5 = {1.0 / 12.0, -2.0 / 3.0, 0.0,
                                                2.0 / 3.0, -1.0 / 12.0};
static const gt::gtensor<double, 1> stencil7 = {
  -1.0 / 60.0, 3.0 / 20.0, -3.0 / 4.0, 0.0, 3.0 / 4.0, -3.0 / 20.0, 1.0 / 60.0};

inline auto stencil1d_3(const gt::gtensor<double, 1>& y,
                        const gt::gtensor<double, 1>& stencil)
{
  return stencil(0) * y.view(_s(0, -2)) + stencil(1) * y.view(_s(1, -1)) +
         stencil(2) * y.view(_s(2, _));
}

inline auto stencil1d_5(const gt::gtensor<double, 1>& y,
                        const gt::gtensor<double, 1>& stencil)
{
  return stencil(0) * y.view(_s(0, -4)) + stencil(1) * y.view(_s(1, -3)) +
         stencil(2) * y.view(_s(2, -2)) + stencil(3) * y.view(_s(3, -1)) +
         stencil(4) * y.view(_s(4, _));
}

inline auto stencil1d_7(const gt::gtensor<double, 1>& y,
                        const gt::gtensor<double, 1>& stencil)
{
  return stencil(0) * y.view(_s(0, -6)) + stencil(1) * y.view(_s(1, -5)) +
         stencil(2) * y.view(_s(2, -4)) + stencil(3) * y.view(_s(3, -3)) +
         stencil(4) * y.view(_s(4, -2)) + stencil(5) * y.view(_s(5, -1)) +
         stencil(6) * y.view(_s(6, _));
}

void test_stencils(int n, double dx, double (*fn)(double),
                   double (*dydx)(double))
{
  // Note: gt::shape(20) is a conveniance function for
  // gt::shape_type<1>(20)
  gt::gtensor<double, 1> x(gt::shape(n));
  gt::gtensor<double, 1> y = gt::empty_like(x);
  gt::gtensor<double, 1> dydx_analytic = gt::empty_like(x);
  gt::gtensor<double, 1> dydx_numeric_3, dydx_numeric_5, dydx_numeric_7;
  for (int i = 0; i < x.shape(0); i++) {
    double xtmp = i * dx;
    x(i) = xtmp;
    y(i) = fn(xtmp);
    dydx_analytic(i) = dydx(xtmp);
  }
  std::cout << "x            = " << x << std::endl;
  std::cout << "y            = " << y << std::endl;
  std::cout << "dydx_analytic = " << dydx_analytic << std::endl;

  std::cout << std::endl;
  dydx_numeric_3 = stencil1d_3(y, stencil3) / dx;
  std::cout << "dydx_numeric3 =   " << dydx_numeric_3 << std::endl;
  std::cout << "err 3         =   "
            << dydx_analytic.view(_s(1, -1)) - dydx_numeric_3 << std::endl;

  std::cout << std::endl;
  dydx_numeric_5 = stencil1d_5(y, stencil5) / dx;
  std::cout << "dydx_numeric5 =     " << dydx_numeric_5 << std::endl;
  std::cout << "err 5         =     "
            << dydx_analytic.view(_s(2, -2)) - dydx_numeric_5 << std::endl;

  std::cout << std::endl;
  dydx_numeric_7 = stencil1d_7(y, stencil7) / dx;
  std::cout << "dydx_numeric7 =     " << dydx_numeric_7 << std::endl;
  std::cout << "err 7         =     "
            << dydx_analytic.view(_s(3, -3)) - dydx_numeric_7 << std::endl;
}

int main(int argc, char** argv)
{
  double dx = 0.1;
  int n = 20;
  std::cout << "y = x^2" << std::endl;
  test_stencils(
    n, dx, [](double x) { return x * x; }, [](double x) { return 2.0 * x; });
  std::cout << std::endl;

  std::cout << "y = x^3" << std::endl;
  test_stencils(
    n, dx, [](double x) { return x * x * x; },
    [](double x) { return 3.0 * x * x; });
  std::cout << std::endl;

  std::cout << "y = x^5" << std::endl;
  test_stencils(
    n, dx, [](double x) { return pow(x, 5); },
    [](double x) { return 5.0 * pow(x, 4); });
  std::cout << std::endl;
}
