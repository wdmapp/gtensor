#include <gtest/gtest.h>

#include <gtensor/gtensor.h>

#include "test_debug.h"

TEST(span, convert_const)
{
  constexpr int N = 1024;
  float a[N];

  for (int i = 0; i < N; i++) {
    a[i] = static_cast<float>(i);
  }
  gt::span<float> sa_mut(&a[0], N);
  gt::span<float> sa_mut_copy(sa_mut);
  gt::span<const float> sa_const(sa_mut);

  // won't compile, different storage size so conversion ctor not defined
  // gt::span<double> sa_double(sa_mut);

  gt::span<const float> sa_const2 = sa_mut;

  EXPECT_EQ(sa_mut.data(), sa_mut_copy.data());
  EXPECT_EQ(&sa_mut[N - 1], &sa_mut_copy[N - 1]);

  EXPECT_EQ(sa_mut.data(), sa_const.data());
  EXPECT_EQ(&sa_mut[N - 1], &sa_const[N - 1]);

  EXPECT_EQ(sa_mut.data(), sa_const2.data());
  EXPECT_EQ(&sa_mut[N - 1], &sa_const2[N - 1]);
}

TEST(span, type_aliases)
{
  constexpr int N = 1024;
  double a[N];
  gt::span<double> h1(&a[0], N);

  GT_DEBUG_TYPE_NAME(decltype(h1)::value_type);
  GT_DEBUG_TYPE_NAME(decltype(h1)::reference);
  GT_DEBUG_TYPE_NAME(decltype(h1)::const_reference);
  GT_DEBUG_TYPE_NAME(decltype(h1)::pointer);
  GT_DEBUG_TYPE_NAME(decltype(h1)::const_pointer);

  EXPECT_TRUE((std::is_same<decltype(h1)::value_type, double>::value));
  EXPECT_TRUE((std::is_same<decltype(h1)::reference, double&>::value));
  EXPECT_TRUE(
    (std::is_same<decltype(h1)::const_reference, const double&>::value));
  EXPECT_TRUE((std::is_same<decltype(h1)::pointer, double*>::value));
  EXPECT_TRUE(
    (std::is_same<decltype(h1)::const_pointer, const double*>::value));
}

#ifdef GTENSOR_HAVE_DEVICE

namespace gt
{
template <typename T>
using device_span = gt::span<
  T, typename gt::space::space_traits<gt::space::device>::template pointer<T>>;
}

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
