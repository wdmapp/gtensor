#include <gtest/gtest.h>

#include <gtensor/gtensor.h>
#include <gtensor/sarray.h>

#include "test_debug.h"

TEST(sarray, construct)
{
  gt::sarray<int, 4> a(1, 2, 3, 4);
  gt::sarray<int, 4> b({1, 2, 3, 4});
  int data[4] = {1, 2, 3, 4};
  gt::sarray<int, 4> c(&data[0], 4);

  EXPECT_EQ(a, b);
  EXPECT_EQ(a, c);
}

template <typename S>
void test_launch_insert()
{
  gt::sarray<int, 5> a(0, 1, 2, 3, 4);

  gt::gtensor<int, 1, S> g2(gt::shape(6));
  gt::gtensor<int, 1, S> g3(gt::shape(7));
  gt::gtensor<int, 1> h2(gt::shape(6));
  gt::gtensor<int, 1> h3(gt::shape(7));

  auto k2 = g2.to_kernel();
  auto k3 = g3.to_kernel();

  gt::launch<1, S>(
    g3.shape(), GT_LAMBDA(int i) {
      auto a2 = insert(a, 0, -1);
      auto a3 = insert(a2, 6, 5);
      if (i < k2.shape(0))
        k2(i) = a2[i];
      k3(i) = a3[i];
    });

  gt::copy(g2, h2);
  EXPECT_EQ(h2, (gt::gtensor<int, 1>{-1, 0, 1, 2, 3, 4}));
  gt::copy(g3, h3);
  EXPECT_EQ(h3, (gt::gtensor<int, 1>{-1, 0, 1, 2, 3, 4, 5}));
}

template <typename S>
void test_launch_remove()
{
  gt::sarray<int, 5> a(0, 1, 2, 3, 4);
  gt::gtensor<int, 1, S> g2(gt::shape(4));
  gt::gtensor<int, 1, S> g3(gt::shape(3));
  gt::gtensor<int, 1> h2(gt::shape(4));
  gt::gtensor<int, 1> h3(gt::shape(3));

  auto k2 = g2.to_kernel();
  auto k3 = g3.to_kernel();

  gt::launch<1, S>(
    g2.shape(), GT_LAMBDA(int i) {
      auto a2 = remove(a, 0);
      auto a3 = remove(a2, 3);
      k2(i) = a2[i];
      if (i < k3.shape(0))
        k3(i) = a3[i];
    });

  gt::copy(g2, h2);
  EXPECT_EQ(h2, (gt::gtensor<int, 1>{1, 2, 3, 4}));
  gt::copy(g3, h3);
  EXPECT_EQ(h3, (gt::gtensor<int, 1>{1, 2, 3}));
}

template <typename S>
void test_launch_assign()
{
  gt::sarray<int, 5> a(0, 1, 2, 3, 4);
  gt::gtensor<int, 1, S> g2(gt::shape(5));
  gt::gtensor<int, 1> h2(gt::shape(5));

  auto k2 = g2.to_kernel();

  gt::launch<1, S>(
    g2.shape(), GT_LAMBDA(int i) {
      gt::sarray<int, 5> a2;
      for (int j = 0; j < 5; j++) {
        a2[j] = 2 * a[j];
      }
      k2(i) = a2[i];
    });

  gt::copy(g2, h2);
  EXPECT_EQ(h2, (gt::gtensor<int, 1>{0, 2, 4, 6, 8}));
}

TEST(sarray, insert)
{
  gt::sarray<int, 5> a(0, 1, 2, 3, 4);

  auto a2 = insert(a, 0, -1);
  GT_DEBUG_TYPE(a2);
  EXPECT_EQ(a2[0], -1);
  EXPECT_EQ(a2[1], 0);

  auto a3 = insert(a2, 6, 5);
  GT_DEBUG_TYPE(a3);
  EXPECT_EQ(a3[0], -1);
  EXPECT_EQ(a3[1], 0);
  EXPECT_EQ(a3[5], 4);
  EXPECT_EQ(a3[6], 5);
}

TEST(sarray, remove)
{
  gt::sarray<int, 5> a(0, 1, 2, 3, 4);

  auto a2 = remove(a, 0);
  GT_DEBUG_TYPE(a2);
  EXPECT_EQ(a2[0], 1);
  EXPECT_EQ(a2[1], 2);

  auto a3 = remove(a2, 3);
  GT_DEBUG_TYPE(a3);
  EXPECT_EQ(a3[0], 1);
  EXPECT_EQ(a3[1], 2);
  EXPECT_EQ(a3[2], 3);
}

TEST(sarray, host_launch_insert)
{
  test_launch_insert<gt::space::host>();
}

TEST(sarray, host_launch_remove)
{
  test_launch_remove<gt::space::host>();
}

TEST(sarray, host_launch_assign)
{
  test_launch_assign<gt::space::host>();
}

#ifdef GTENSOR_HAVE_DEVICE

TEST(sarray, device_launch_insert)
{
  test_launch_insert<gt::space::device>();
}

TEST(sarray, device_launch_remove)
{
  test_launch_remove<gt::space::device>();
}

TEST(sarray, device_launch_assign)
{
  test_launch_assign<gt::space::device>();
}

#endif
