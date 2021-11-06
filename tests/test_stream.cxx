#include <gtest/gtest.h>

#include "gtensor/gtensor.h"

#include "test_debug.h"

TEST(stream, assign_gtensor_6d)
{
  gt::gtensor<int, 6> a(gt::shape(2, 3, 4, 5, 6, 7));
  gt::gtensor<int, 6> b(a.shape());

  gt::stream stream;

  int* adata = a.data();

  for (int i = 0; i < a.size(); i++) {
    adata[i] = i;
  }

  EXPECT_NE(a, b);
  gt::assign(b, a, stream.get_view());
  stream.synchronize();
  EXPECT_EQ(a, b);
}

void device_double_add_2d_stream(gt::gtensor_device<double, 2>& a,
                                 gt::gtensor<double, 2>& out,
                                 gt::stream_view stream)
{
  auto b = gt::empty_like(a);

  auto k_a = a.to_kernel();
  auto k_b = b.to_kernel();

  gt::launch<2>(
    a.shape(), GT_LAMBDA(int i, int j) { k_b(i, j) = k_a(i, j) + k_a(i, j); },
    stream);
  gt::copy(b, out);
}

TEST(stream, stream_device_launch_2d)
{
  gt::gtensor_device<double, 2> a{{11., 12., 13.}, {21., 22., 23.}};
  gt::gtensor<double, 2> h_b(a.shape());

  gt::stream stream;

  device_double_add_2d_stream(a, h_b, stream.get_view());

  EXPECT_EQ(h_b, (gt::gtensor<double, 2>{{22., 24., 26.}, {42., 44., 46.}}));
}
