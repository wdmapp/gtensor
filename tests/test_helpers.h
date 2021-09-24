#include <iostream>

#include <gtensor/complex.h>

#include "test_debug.h"

namespace gt
{

namespace test
{

namespace detail
{
template <typename Real>
struct max_err;

template <>
struct max_err<double>
{
  static constexpr double value = 1e-14;
};

template <>
struct max_err<float>
{
  static constexpr double value = 1e-5;
};

template <>
struct max_err<gt::complex<double>>
{
  static constexpr double value = 1e-14;
};

template <>
struct max_err<gt::complex<float>>
{
  static constexpr double value = 1e-5;
};

} // namespace detail

} // namespace test

} // namespace gt

template <typename Real>
inline void expect_complex_near(gt::complex<Real> x, gt::complex<Real> y,
                                double max_err = -1.0)
{
  if (max_err == -1.0) {
    max_err = gt::test::detail::max_err<Real>::value;
  }
  EXPECT_NEAR(x.real(), y.real(), max_err);
  EXPECT_NEAR(x.imag(), y.imag(), max_err);
}

template <typename Real>
inline void expect_complex_near(gt::complex<Real> x, double y,
                                double max_err = -1.0)
{
  if (max_err == -1.0) {
    max_err = gt::test::detail::max_err<Real>::value;
  }
  EXPECT_NEAR(x.real(), y, max_err);
  EXPECT_NEAR(x.imag(), 0.0, max_err);
}

template <typename E1, typename E2>
inline void _expect_near_array(const char* file, int line, const char* xname,
                               E1 x, const char* yname, E2 y,
                               double max_err = -1.0)
{
  using V = gt::expr_value_type<E1>;
  bool equal = true;
  int i;
  double err;

  if (max_err == -1.0) {
    max_err = gt::test::detail::max_err<V>::value;
  }
  auto xflat = gt::flatten(x);
  auto yflat = gt::flatten(y);

  for (i = 0; i < xflat.shape(0); i++) {
    err = gt::abs(xflat(i) - yflat(i));
    if (err > max_err) {
      equal = false;
      break;
    }
  }

  if (!equal) {
    std::cerr << "Arrays not close (max " << max_err << ") at " << file << ":"
              << line << std::endl
              << " err " << err << " at [" << i << "]" << std::endl
              << " " << xname << ":" << std::endl
              << x << std::endl
              << " " << yname << ":" << std::endl
              << y << std::endl;
  }
  EXPECT_TRUE(equal);
}

#define GT_EXPECT_NEAR_ARRAY(x, y)                                             \
  _expect_near_array(__FILE__, __LINE__, #x, x, #y, y)
