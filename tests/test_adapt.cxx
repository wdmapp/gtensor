#include <gtest/gtest.h>

#include "gtensor/gtensor.h"

#include "test_debug.h"

using namespace gt::placeholders;

TEST(adapt, adapt_complex)
{
  constexpr int N = 8;
  constexpr int M = 16;
  constexpr int S = N * M;
  using T = gt::complex<double>;
  T* a = gt::backend::standard::host_allocator<T>{}.allocate(S);
  auto gta = gt::adapt<2>(a, gt::shape(N, M));

  GT_DEBUG_TYPE(gta);
  // GT_DEBUG_TYPE_NAME(decltype(gta)::base_type);

  auto gta_view = gta.view(_all, _all);
  GT_DEBUG_TYPE(gta_view);
  // GT_DEBUG_TYPE_NAME(decltype(gta_view)::base_type);

  gta(0, 0) = T{1., 1.};
  gta_view(N - 1, M - 1) = T{1., -1.};

  EXPECT_EQ(a[0], (T{1., 1.}));
  EXPECT_EQ(a[S - 1], (T{1., -1.}));

  gt::backend::standard::host_allocator<T>{}.deallocate(a, S);
}

#ifdef GTENSOR_HAVE_DEVICE

TEST(adapt, adapt_device)
{
  constexpr int N = 10;
  int* a = gt::backend::standard::device_allocator<int>{}.allocate(N);
  auto aview = gt::adapt_device(a, gt::shape(N));

  aview = gt::scalar(7);

  gt::gtensor<int, 1> h_a{gt::shape(N)};

  gt::copy(aview, h_a);

  gt::gtensor<int, 1> h_expected{7, 7, 7, 7, 7, 7, 7, 7, 7, 7};

  EXPECT_EQ(aview, h_expected);

  gt::backend::standard::device_allocator<int>{}.deallocate(a, N);
}

#endif // GTENSOR_HAVE_DEVICE
