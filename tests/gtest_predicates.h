#include <algorithm>
#include <iostream>
#include <type_traits>

#include <gtensor/gtensor.h>

#include <gtest/gtest.h>

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

template <typename R1, typename R2>
std::enable_if_t<std::is_arithmetic<R1>::value && std::is_arithmetic<R2>::value,
                 testing::AssertionResult>
pred_near3(const char* xname, const char* yname, const char* mname, R1 x, R2 y,
           double max_err = -1.0)
{
  using R = decltype(x - y);
  if (max_err == -1.0) {
    max_err = gt::test::detail::max_err<R>::value;
  }
  R actual_err = std::abs(x - y);
  if (actual_err > max_err) {
    auto af = testing::AssertionFailure();
    if (max_err == 0) {
      af << "Expected equal values:\n";
    } else {
      af << "Expected near values (err=" << actual_err << " > " << max_err
         << "):\n";
    }
    return af << "  " << xname << "\n    Which is: " << x << "\n"
              << "  " << yname << "\n    Which is: " << y << "\n";
  }
  return testing::AssertionSuccess();
}

template <typename R1, typename R2>
std::enable_if_t<std::is_floating_point<R1>::value &&
                   std::is_floating_point<R2>::value,
                 testing::AssertionResult>
pred_near3(const char* xname, const char* yname, const char* mname,
           gt::complex<R1> x, gt::complex<R2> y, double max_err = -1.0)
{
  using R = decltype(std::declval<R1>() - std::declval<R2>());
  if (max_err == -1.0) {
    max_err = gt::test::detail::max_err<R>::value;
  }
  double actual_err = gt::abs(x - y);
  if (actual_err > max_err) {
    auto af = testing::AssertionFailure();
    if (max_err == 0) {
      af << "Expected equal values:\n";
    } else {
      af << "Expected near values (err=" << actual_err << " > " << max_err
         << "):\n";
    }
    return af << "  " << xname << "\n    Which is: " << x << "\n"
              << "  " << yname << "\n    Which is: " << y << "\n";
  }
  return testing::AssertionSuccess();
}

template <typename R1, typename R2>
std::enable_if_t<std::is_floating_point<R1>::value &&
                   std::is_arithmetic<R2>::value,
                 testing::AssertionResult>
pred_near3(const char* xname, const char* yname, const char* mname,
           gt::complex<R1> x, R2 y, double max_err = -1.0)
{
  return pred_near3(xname, yname, mname,
                    gt::complex<double>(x.real(), x.imag()),
                    gt::complex<double>(y, 0.0), max_err);
}

template <typename E1, typename E2>
std::enable_if_t<gt::is_expression<E1>::value && gt::is_expression<E2>::value,
                 testing::AssertionResult>
pred_near3(const char* xname, const char* yname, const char* mname, E1&& x,
           E2&& y, double max_err = -1.0)
{
  using V = gt::expr_value_type<E1>;
  bool equal = true;
  int i;
  double actual_err;

  if (max_err == -1.0) {
    max_err = gt::test::detail::max_err<V>::value;
  }
  auto xflat = gt::flatten(x);
  auto yflat = gt::flatten(y);

  assert(xflat.shape(0) == yflat.shape(0));

  int n = xflat.shape(0);

  for (i = 0; i < n; i++) {
    actual_err = gt::abs(xflat(i) - yflat(i));
    if (actual_err > max_err) {
      equal = false;
      break;
    }
  }

  if (!equal) {
    const int max_view = 10;
    int start = i;
    int tail = start + max_view - n;
    if (tail > 0) {
      start = std::max(start - tail, 0);
    }
    int end = std::min(start + max_view, n);
    auto xs = gt::slice(start, end);

    auto af = testing::AssertionFailure();
    if (max_err == 0) {
      af << "Expected equal values";
    } else {
      af << "Expected near values (err=" << actual_err << " > " << max_err
         << ")";
    }
    af << " at [" << i << "]:\n";
    return af << "  " << xname << "\n    Which is: " << xflat.view(xs) << "\n"
              << "  " << yname << "\n    Which is: " << yflat.view(xs) << "\n";
  }

  return testing::AssertionSuccess();
}

template <typename E1, typename R>
std::enable_if_t<gt::is_expression<E1>::value && !gt::is_expression<R>::value,
                 testing::AssertionResult>
pred_near3(const char* xname, const char* yname, const char* mname, E1&& x, R y,
           double max_err = -1.0)
{
  using V = gt::expr_value_type<E1>;
  bool equal = true;
  int i, n;
  double actual_err;

  if (max_err == -1.0) {
    max_err = gt::test::detail::max_err<V>::value;
  }

  auto xflat = gt::flatten(x);
  n = xflat.shape(0);

  for (i = 0; i < n; i++) {
    actual_err = std::abs(gt::abs(xflat(i)) - y);
    if (actual_err > max_err) {
      equal = false;
      break;
    }
  }

  if (!equal) {
    const int max_view = 10;
    int start = i;
    int tail = start + max_view - n;
    if (tail > 0) {
      start = std::max(start - tail, 0);
    }
    int end = std::min(start + max_view, n);
    auto xs = gt::slice(start, end);

    auto af = testing::AssertionFailure();
    if (max_err == 0) {
      af << "Expected equal values";
    } else {
      af << "Expected near values (err=" << actual_err << " > " << max_err
         << ")";
    }
    af << " at [" << i << "]:\n";
    return af << "  " << xname << "\n    Which is: " << xflat.view(xs) << "\n"
              << "  " << yname << "\n    Which is: " << y << "\n";
  }

  return testing::AssertionSuccess();
}

} // namespace test

} // namespace gt

#define GT_EXPECT_NEAR(a, b)                                                   \
  EXPECT_PRED_FORMAT3(gt::test::pred_near3, a, b, -1.0)

#define GT_EXPECT_NEAR_MAXERR(a, b, max)                                       \
  EXPECT_PRED_FORMAT3(gt::test::pred_near3, a, b, static_cast<double>(max))

#define GT_EXPECT_EQ(a, b) EXPECT_PRED_FORMAT3(gt::test::pred_near3, a, b, 0.0)
