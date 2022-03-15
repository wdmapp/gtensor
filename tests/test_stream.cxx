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

#ifdef GTENSOR_HAVE_DEVICE

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

// Compile with NDEBUG _NOT_ set, to make sure gpuSyncIfEnabled macro
// is building correctly
TEST(stream, assign_sync_if_enabled)
{
  gt::gtensor_device<double, 2> a{{11., 12., 13.}, {21., 22., 23.}};
  gt::gtensor_device<double, 2> b(a.shape());
  gt::gtensor<double, 2> h_a(a.shape());
  auto h_b = gt::zeros<double>(a.shape());

  double* adata = h_a.data();

  for (int i = 0; i < a.size(); i++) {
    adata[i] = i;
  }

  gt::copy(h_a, a);

  EXPECT_NE(h_a, h_b);
  gt::assign(b, a);
  gpuSyncIfEnabled();

  gt::copy(b, h_b);
  EXPECT_EQ(h_a, h_b);
}

TEST(stream, device_assign_ordered_default)
{
  gt::gtensor_device<double, 2> a{{11., 12., 13.}, {21., 22., 23.}};
  gt::gtensor_device<double, 2> b(a.shape());
  gt::gtensor_device<double, 2> c(a.shape());
  auto h_a = gt::empty<double>(a.shape());
  auto h_b = gt::zeros<double>(a.shape());
  auto h_c = gt::zeros<double>(a.shape());

  double* adata = h_a.data();

  for (int i = 0; i < a.size(); i++) {
    adata[i] = i;
  }

  gt::copy(h_a, a);

  EXPECT_NE(h_a, h_b);
  gt::assign(b, a);
  gt::assign(c, b);

  gt::synchronize();

  gt::copy(b, h_b);
  EXPECT_EQ(h_b, h_a);

  gt::copy(c, h_c);
  EXPECT_EQ(h_c, h_a);
}

TEST(stream, device_assign_ordered_non_default)
{
  gt::gtensor_device<double, 2> a{{11., 12., 13.}, {21., 22., 23.}};
  gt::gtensor_device<double, 2> b(a.shape());
  gt::gtensor_device<double, 2> c(a.shape());
  auto h_a = gt::empty<double>(a.shape());
  auto h_b = gt::zeros<double>(a.shape());
  auto h_c = gt::zeros<double>(a.shape());

  gt::stream stream;

  double* adata = h_a.data();

  for (int i = 0; i < a.size(); i++) {
    adata[i] = i;
  }

  gt::copy(h_a, a);

  EXPECT_NE(h_a, h_b);
  gt::assign(b, a, stream.get_view());
  gt::assign(c, b, stream.get_view());

  stream.synchronize();

  gt::copy(b, h_b);
  EXPECT_EQ(h_b, h_a);

  gt::copy(c, h_c);
  EXPECT_EQ(h_c, h_a);
}

#endif
