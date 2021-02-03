#include <gtest/gtest.h>

#include <gtensor/gtensor.h>

#include <type_traits>

#include "test_debug.h"

#ifdef GTENSOR_USE_THRUST

template <typename S>
void test_sum(int n)
{
  gt::gtensor<double, 1> a(gt::shape(n));
  for (int i = 0; i < n; i++) {
    a(i) = i + 1;
  }
  double asum = 0.0;
  if (std::is_same<S, gt::space::host>::value) {
    asum = gt::sum(a);
  } else {
    gt::gtensor<double, 1, S> a2(gt::shape(n));
    gt::copy(a, a2);
    asum = gt::sum(a2);
  }
  EXPECT_EQ(asum, (double)n * (n + 1) / 2);
}

template <typename S>
void test_max(int n)
{
  gt::gtensor<double, 1> a(gt::shape(n));
  for (int i = 0; i < n; i++) {
    a(i) = i + 1;
  }
  double amax = 0.0;
  if (std::is_same<S, gt::space::host>::value) {
    amax = gt::max(a);
  } else {
    gt::gtensor<double, 1, S> a2(gt::shape(n));
    gt::copy(a, a2);
    amax = gt::max(a2);
  }
  EXPECT_EQ(amax, n);
}

template <typename S>
void test_min(int n)
{
  gt::gtensor<double, 1> a(gt::shape(n));
  for (int i = 0; i < n; i++) {
    a(i) = n - i;
  }
  double amin = 0.0;
  if (std::is_same<S, gt::space::host>::value) {
    amin = gt::min(a);
  } else {
    gt::gtensor<double, 1, S> a2(gt::shape(n));
    gt::copy(a, a2);
    amin = gt::min(a2);
  }
  EXPECT_EQ(amin, 1);
}

TEST(gtensor, sum_1d)
{
  test_sum<gt::space::host>(2048);
}

TEST(gtensor, max_1d)
{
  test_max<gt::space::host>(2048);
}

TEST(gtensor, min_1d)
{
  test_min<gt::space::host>(2048);
}

#ifdef GTENSOR_HAVE_DEVICE

TEST(gtensor, device_sum_1d)
{
  test_sum<gt::space::device>(2048);
}

TEST(gtensor, device_max_1d)
{
  test_max<gt::space::device>(2048);
}

TEST(gtensor, device_min_1d)
{
  test_min<gt::space::device>(2048);
}

#endif // GTENSOR_HAVE_DEVICE

#endif // GTENSOR_USE_THRUST
