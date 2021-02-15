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

TEST(assign, gtensor_fill)
{
  gt::gtensor<float, 2> a(gt::shape(2, 3));

  a.fill(9001);

  EXPECT_EQ(a,
            (gt::gtensor<float, 2>{{9001, 9001}, {9001, 9001}, {9001, 9001}}));
}

TEST(assign, gview_fill)
{
  gt::gtensor<float, 2> a(gt::shape(2, 3));
  auto av = a.view(gt::all, gt::all);

  av.fill(9001);

  EXPECT_EQ(a,
            (gt::gtensor<float, 2>{{9001, 9001}, {9001, 9001}, {9001, 9001}}));
}

TEST(assign, span_fill)
{
  gt::gtensor<float, 2> a(gt::shape(2, 3));
  auto as = a.to_kernel();

  as.fill(9001);

  EXPECT_EQ(a,
            (gt::gtensor<float, 2>{{9001, 9001}, {9001, 9001}, {9001, 9001}}));
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

TEST(assign, device_gtensor_fill)
{
  gt::gtensor_device<float, 2> a(gt::shape(2, 3));
  gt::gtensor<float, 2> h_a(a.shape());

  a.fill(9001);

  gt::copy(a, h_a);
  EXPECT_EQ(h_a, (gt::gtensor_device<float, 2>{
                   {9001, 9001}, {9001, 9001}, {9001, 9001}}));
}

TEST(assign, device_gview_fill)
{
  gt::gtensor_device<float, 2> a(gt::shape(2, 3));
  gt::gtensor<float, 2> h_a(a.shape());
  auto av = a.view(gt::all, gt::all);

  av.fill(9001);

  gt::copy(a, h_a);
  EXPECT_EQ(h_a,
            (gt::gtensor<float, 2>{{9001, 9001}, {9001, 9001}, {9001, 9001}}));
}

TEST(assign, device_span_fill)
{
  gt::gtensor_device<float, 2> a(gt::shape(2, 3));
  gt::gtensor<float, 2> h_a(a.shape());
  auto as = a.to_kernel();

  as.fill(9001);

  gt::copy(a, h_a);
  EXPECT_EQ(h_a,
            (gt::gtensor<float, 2>{{9001, 9001}, {9001, 9001}, {9001, 9001}}));
}

#endif // GTENSOR_HAVE_DEVICE
