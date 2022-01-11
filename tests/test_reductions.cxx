#include <gtest/gtest.h>

#include <gtensor/gtensor.h>
#include <gtensor/reductions.h>

#include <functional>
#include <type_traits>

#include "test_debug.h"

using namespace gt::placeholders;

TEST(reductions, sum_axis_to_2d)
{
  gt::gtensor<double, 2> a({{11., 21., 31.}, {12., 22., 32.}});
  GT_DEBUG_VAR(a.shape());

  gt::gtensor<double, 1> asum0(gt::shape(2));
  gt::gtensor<double, 1> asum1(gt::shape(3));

  sum_axis_to(asum0, a, 0);
  EXPECT_EQ(asum0, (gt::gtensor<double, 1>{63., 66.}));
  sum_axis_to(asum1, a, 1);
  EXPECT_EQ(asum1, (gt::gtensor<double, 1>{23., 43., 63.}));
}

TEST(reductions, sum_axis_to_2d_view)
{
  gt::gtensor<double, 2> a({{11., 21., 31.}, {12., 22., 32.}});

  gt::gtensor<double, 1> asum0(gt::shape(2));
  gt::gtensor<double, 1> asum1(gt::shape(3));

  sum_axis_to(asum0.view(_all), a.view(_all, _all), 0);
  EXPECT_EQ(asum0, (gt::gtensor<double, 1>{63., 66.}));
  sum_axis_to(asum1.view(_all), a.view(_all, _all), 1);
  EXPECT_EQ(asum1, (gt::gtensor<double, 1>{23., 43., 63.}));
}

TEST(reductions, sum_axis_to_3d_view_2d)
{
  gt::gtensor<double, 3> a({{{11., 21., 31.}, {12., 22., 32.}},
                            {{-11., -21., -31.}, {-12., -22., -32.}}});

  EXPECT_EQ(a.view(_all, _all, 0),
            (gt::gtensor<double, 2>{{11., 21., 31.}, {12., 22., 32.}}));
  EXPECT_EQ(a.view(_all, _all, 1),
            (gt::gtensor<double, 2>{{-11., -21., -31.}, {-12., -22., -32.}}));

  gt::gtensor<double, 2> asum0(gt::shape(2, 2));
  gt::gtensor<double, 2> asum1(gt::shape(2, 3));

  sum_axis_to(asum0.view(0, _all), a.view(_all, _all, 0), 0);
  sum_axis_to(asum0.view(1, _all), a.view(_all, _all, 1), 0);
  EXPECT_EQ(asum0.view(0, _all), (gt::gtensor<double, 1>{63., 66.}));
  EXPECT_EQ(asum0.view(1, _all), (gt::gtensor<double, 1>{-63., -66.}));

  sum_axis_to(asum1.view(0, _all), a.view(_all, _all, 0), 1);
  sum_axis_to(asum1.view(1, _all), a.view(_all, _all, 1), 1);
  EXPECT_EQ(asum1.view(0, _all), (gt::gtensor<double, 1>{23., 43., 63.}));
  EXPECT_EQ(asum1.view(1, _all), (gt::gtensor<double, 1>{-23., -43., -63.}));
}

#ifdef GTENSOR_HAVE_DEVICE

TEST(reductions, device_sum_axis_to_2d)
{
  gt::gtensor_device<double, 2> a({{11., 21., 31.}, {12., 22., 32.}});
  GT_DEBUG_VAR(a.shape());

  gt::gtensor_device<double, 1> asum0(gt::shape(2));
  gt::gtensor_device<double, 1> asum1(gt::shape(3));
  gt::gtensor<double, 1> h_asum0(gt::shape(2));
  gt::gtensor<double, 1> h_asum1(gt::shape(3));

  sum_axis_to(asum0, a, 0);
  gt::copy(asum0, h_asum0);
  EXPECT_EQ(h_asum0, (gt::gtensor<double, 1>{63., 66.}));
  sum_axis_to(asum1, a, 1);
  gt::copy(asum1, h_asum1);
  EXPECT_EQ(h_asum1, (gt::gtensor<double, 1>{23., 43., 63.}));
}

TEST(reductions, device_sum_axis_to_3d_view_2d)
{
  gt::gtensor_device<double, 3> a({{{11., 21., 31.}, {12., 22., 32.}},
                                   {{-11., -21., -31.}, {-12., -22., -32.}}});

  EXPECT_EQ(a.view(_all, _all, 0),
            (gt::gtensor<double, 2>{{11., 21., 31.}, {12., 22., 32.}}));
  EXPECT_EQ(a.view(_all, _all, 1),
            (gt::gtensor<double, 2>{{-11., -21., -31.}, {-12., -22., -32.}}));

  gt::gtensor_device<double, 2> asum0(gt::shape(2, 2));
  gt::gtensor_device<double, 2> asum1(gt::shape(2, 3));
  gt::gtensor<double, 2> h_asum0(asum0.shape());
  gt::gtensor<double, 2> h_asum1(asum1.shape());

  sum_axis_to(asum0.view(0, _all), a.view(_all, _all, 0), 0);
  sum_axis_to(asum0.view(1, _all), a.view(_all, _all, 1), 0);
  gt::copy(asum0, h_asum0);
  EXPECT_EQ(h_asum0.view(0, _all), (gt::gtensor<double, 1>{63., 66.}));
  EXPECT_EQ(h_asum0.view(1, _all), (gt::gtensor<double, 1>{-63., -66.}));

  sum_axis_to(asum1.view(0, _all), a.view(_all, _all, 0), 1);
  sum_axis_to(asum1.view(1, _all), a.view(_all, _all, 1), 1);
  gt::copy(asum1, h_asum1);
  EXPECT_EQ(h_asum1.view(0, _all), (gt::gtensor<double, 1>{23., 43., 63.}));
  EXPECT_EQ(h_asum1.view(1, _all), (gt::gtensor<double, 1>{-23., -43., -63.}));
}

#endif // GTENSOR_HAVE_DEVICE

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

TEST(reductions, sum_1d)
{
  test_sum<gt::space::host>(2048);
}

TEST(reductions, max_1d)
{
  test_max<gt::space::host>(2048);
}

TEST(reductions, min_1d)
{
  test_min<gt::space::host>(2048);
}

TEST(reductions, norm)
{
  gt::gtensor<double, 1> g = {1., -3., 2.};
  EXPECT_EQ(gt::norm_linf(g), 3.);
}

TEST(reductions, norm_expr)
{
  gt::gtensor<double, 1> g = {1., -3., 2.};
  EXPECT_EQ(gt::norm_linf(2. * g), 6.);
}

#ifdef GTENSOR_HAVE_DEVICE

TEST(reductions, device_sum_1d)
{
  test_sum<gt::space::device>(2048);
}

TEST(reductions, device_max_1d)
{
  test_max<gt::space::device>(2048);
}

TEST(reductions, device_min_1d)
{
  test_min<gt::space::device>(2048);
}

#endif // GTENSOR_HAVE_DEVICE

template <typename S, typename T>
void test_reduce_sum(int n)
{
  gt::gtensor<T, 1> a(gt::shape(n));
  for (int i = 0; i < n; i++) {
    a(i) = i + 1;
  }
  T asum = 0.0;
  if (std::is_same<S, gt::space::host>::value) {
    asum = gt::reduce(a, (T)0.0, std::plus<T>{});
  } else {
    gt::gtensor<T, 1, S> a2(gt::shape(n));
    gt::copy(a, a2);
    asum = gt::reduce(a2, (T)0.0, std::plus<T>{});
  }
  EXPECT_EQ(asum, (T)n * (n + 1) / 2);
}

TEST(reductions, reduce_sum_1d)
{
  test_reduce_sum<gt::space::host, double>(2048);
}

#ifdef GTENSOR_HAVE_DEVICE

TEST(reductions, device_reduce_sum_1d)
{
  test_reduce_sum<gt::space::device, double>(2048);
}

#endif // GTENSOR_HAVE_DEVICE

template <typename Tin, typename Tout = Tin, typename Enable = void>
struct UnaryOpNorm
{
  GT_INLINE Tout operator()(Tin a) const { return a * a; }
};

template <typename Tin, typename Tout>
struct UnaryOpNorm<Tin, Tout,
                   gt::meta::void_t<std::enable_if_t<gt::is_complex_v<Tin>>>>
{
  GT_INLINE Tout operator()(Tin a) const { return gt::norm(a); }
};

template <typename S, typename T>
void test_transform_reduce_sum_sq(int n)
{
  using Real = gt::complex_subtype_t<T>;
  gt::gtensor<T, 1> a(gt::shape(n));
  for (int i = 0; i < n; i++) {
    a(i) = i + 1;
  }
  T asum = 0.0;
  if (std::is_same<S, gt::space::host>::value) {
    asum = gt::transform_reduce(a, 0.0, std::plus<>{}, UnaryOpNorm<T, Real>{});
  } else {
    gt::gtensor<T, 1, S> a2(gt::shape(n));
    gt::copy(a, a2);
    asum = gt::transform_reduce(a2, 0.0, std::plus<>{}, UnaryOpNorm<T, Real>{});
  }
  EXPECT_EQ(asum, static_cast<Real>(n) * (n + 1) * (2 * n + 1) / 6);
}

TEST(reductions, reduce_sum_sq_1d)
{
  test_transform_reduce_sum_sq<gt::space::host, double>(2048);
}

TEST(reductions, reduce_sum_sq_1d_comlex)
{
  test_transform_reduce_sum_sq<gt::space::host, gt::complex<double>>(2048);
}

#ifdef GTENSOR_HAVE_DEVICE

TEST(reductions, device_reduce_sum_sq_1d)
{
  test_transform_reduce_sum_sq<gt::space::device, double>(2048);
}

TEST(reductions, device_reduce_sum_sq_1d_complex)
{
  test_transform_reduce_sum_sq<gt::space::device, gt::complex<double>>(2048);
}

#endif // GTENSOR_HAVE_DEVICE
