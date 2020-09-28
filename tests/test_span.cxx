#include <gtest/gtest.h>

#include <gtensor/gtensor.h>

#include "test_debug.h"

TEST(span, convert_const)
{
  constexpr int N = 1024;
  double a[N];

  for (int i = 0; i < N; i++) {
    a[i] = static_cast<double>(i);
  }
  gt::span<double> sa_mut(&a[0], N);
  gt::span<double> sa_mut_copy(sa_mut);
  gt::span<const double> sa_const(sa_mut);

  gt::span<const double> sa_const2 = sa_mut;

  EXPECT_EQ(sa_mut.data(), sa_mut_copy.data());
  EXPECT_EQ(&sa_mut[N - 1], &sa_mut_copy[N - 1]);

  EXPECT_EQ(sa_mut.data(), sa_const.data());
  EXPECT_EQ(&sa_mut[N - 1], &sa_const[N - 1]);

  EXPECT_EQ(sa_mut.data(), sa_const2.data());
  EXPECT_EQ(&sa_mut[N - 1], &sa_const2[N - 1]);
}

#ifdef GTENSOR_HAVE_DEVICE

TEST(span, device_convert_const)
{
  constexpr int N = 1024;
  gt::gtensor_device<double, 1> a(gt::shape(N));

  gt::device_span<double> sa_mut(a.data(), N);
  gt::device_span<double> sa_mut_copy(sa_mut);
  gt::device_span<const double> sa_const(sa_mut);

  gt::device_span<const double> sa_const2 = sa_mut;

  EXPECT_EQ(sa_mut.data(), sa_mut_copy.data());
  EXPECT_EQ(&sa_mut[N - 1], &sa_mut_copy[N - 1]);

  EXPECT_EQ(sa_mut.data(), sa_const.data());
  EXPECT_EQ(&sa_mut[N - 1], &sa_const[N - 1]);

  EXPECT_EQ(sa_mut.data(), sa_const2.data());
  EXPECT_EQ(&sa_mut[N - 1], &sa_const2[N - 1]);
}

#endif // GTENSOR_HAVE_DEVICE
