#include <gtest/gtest.h>

#include "gtensor/gtensor.h"

#include "test_debug.h"

using namespace gt::placeholders;

TEST(adapt, adapt_complex)
{
  constexpr int N = 8;
  constexpr int M = 16;
  using T = gt::complex<double>;
  gt::backend::host_storage<T> a(N * M);
  auto gta = gt::adapt<2>(a.data(), gt::shape(N, M));

  GT_DEBUG_TYPE(gta);
  // GT_DEBUG_TYPE_NAME(decltype(gta)::base_type);

  auto gta_view = gta.view(_all, _all);
  GT_DEBUG_TYPE(gta_view);
  // GT_DEBUG_TYPE_NAME(decltype(gta_view)::base_type);

  gta(0, 0) = T{1., 1.};
  gta_view(N - 1, M - 1) = T{1., -1.};

  EXPECT_EQ(a[0], (T{1., 1.}));
  EXPECT_EQ(a[N * M - 1], (T{1., -1.}));
}

TEST(adapt, adapt_copy)
{
  constexpr int N = 3;

  // managed allocation adapted
  auto p_coeff =
    gt::backend::gallocator<gt::space::clib_managed>::allocate<double>(N);
  auto coeff_adapt = gt::adapt<1>(p_coeff, gt::shape(N));

  // host allocation
  gt::gtensor<double, 1> coeff_gt(gt::shape(N));

  // copy between managed and host
  gt::copy(coeff_adapt, coeff_gt);

  EXPECT_EQ(coeff_gt, coeff_adapt);

  // clean up
  gt::backend::gallocator<gt::space::clib_managed>::deallocate<double>(p_coeff);
}

TEST(adapt, adapt_device)
{
  constexpr int N = 10;
  gt::backend::device_storage<int> a(N);
  auto aview = gt::adapt_device(gt::raw_pointer_cast(a.data()), gt::shape(N));

  aview = gt::scalar(7);

  gt::gtensor<int, 1> h_a{gt::shape(N)};

  gt::copy(aview, h_a);

  gt::gtensor<int, 1> h_expected{7, 7, 7, 7, 7, 7, 7, 7, 7, 7};

  EXPECT_EQ(aview, h_expected);
}

TEST(adapt, adapt_device_1d)
{
  constexpr int N = 10;
  gt::backend::device_storage<int> a(N);
  auto aview = gt::adapt_device<1>(gt::raw_pointer_cast(a.data()), {N});

  aview = gt::scalar(7);

  gt::gtensor<int, 1> h_a{gt::shape(N)};

  gt::copy(aview, h_a);

  gt::gtensor<int, 1> h_expected{7, 7, 7, 7, 7, 7, 7, 7, 7, 7};

  EXPECT_EQ(aview, h_expected);
}

TEST(adapt, adapt_device_2d)
{
  constexpr int N = 4;
  constexpr int M = 2;
  gt::backend::device_storage<int> a(N * M);
  auto aview = gt::adapt_device<2>(gt::raw_pointer_cast(a.data()), {N, M});

  aview = gt::scalar(7);

  gt::gtensor<int, 2> h_a{gt::shape(N, M)};

  gt::copy(aview, h_a);

  gt::gtensor<int, 2> h_expected{{7, 7, 7, 7}, {7, 7, 7, 7}};

  EXPECT_EQ(aview, h_expected);
}
