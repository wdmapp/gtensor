#include <gtest/gtest.h>

#include "gtensor/gtensor.h"

#include "test_debug.h"

TEST(assign, gtensor_6d)
{
  gt::gtensor<int, 6> a(gt::shape(2, 3, 4, 5, 6, 7));
  gt::gtensor<int, 6> b(a.shape());

  int* adata = a.data();

  for (int i = 0; i < a.size(); i++) {
    adata[i] = i;
  }

  EXPECT_NE(a, b);
  b = a;
  EXPECT_EQ(a, b);
}

#ifdef GTENSOR_HAVE_DEVICE

TEST(assign, device_gtensor_6d)
{
  gt::gtensor_device<int, 6> a(gt::shape(2, 3, 4, 5, 6, 7));
  gt::gtensor_device<int, 6> b(a.shape());
  gt::gtensor<int, 6> h_a(a.shape());
  gt::gtensor<int, 6> h_b(a.shape());

  int* adata = h_a.data();

  for (int i = 0; i < a.size(); i++) {
    adata[i] = i;
  }

  gt::copy(h_a, a);
  b = a;
  gt::copy(b, h_b);

  EXPECT_EQ(h_a, h_b);
}

#endif // GTENSOR_HAVE_DEVICE
