#include <gtest/gtest.h>

#include "gtensor/gtensor.h"
#include "gtensor/device_backend.h"

#ifdef GTENSOR_HAVE_DEVICE

TEST(adapt, adapt_device)
{
  constexpr int N = 10;
  int *a = gt::backend::device_allocator<int>::allocate(N);
  auto aview = gt::adapt_device(a, gt::shape(N));

  aview = gt::scalar(7);

  gt::gtensor<int, 1> h_a{gt::shape(N)};

  gt::copy(aview, h_a);

  gt::gtensor<int, 1> h_expected{7,7,7,7,7,7,7,7,7,7};

  EXPECT_EQ(aview, h_expected);

  gt::backend::device_allocator<int>::deallocate(a);
}

#endif // GTENSOR_HAVE_DEVICE

