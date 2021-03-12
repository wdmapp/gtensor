#include <gtensor/complex.h>

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

} // namespace detail

} // namespace test

} // namespace gt

template <typename Real>
inline void expect_complex_near(gt::complex<Real> x, gt::complex<Real> y)
{
  double max_err = gt::test::detail::max_err<Real>::value;
  EXPECT_NEAR(x.real(), y.real(), max_err);
  EXPECT_NEAR(x.imag(), y.imag(), max_err);
}

template <typename Real>
inline void expect_complex_near(gt::complex<Real> x, double y)
{
  double max_err = gt::test::detail::max_err<Real>::value;
  EXPECT_NEAR(x.real(), y, max_err);
  EXPECT_NEAR(x.imag(), 0.0, max_err);
}
