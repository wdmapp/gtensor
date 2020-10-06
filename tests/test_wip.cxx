#include <gtest/gtest.h>

#include <gtensor/gtensor.h>

TEST(gview, newaxis)
{
  gt::gtensor<double, 2> a = {{11., 12., 13.}, {21., 22., 23.}};
  gt::gtensor<double, 1> b = {100., 101., 102.};

  auto bv = gt::view<2>(b, {gt::all, gt::newaxis});
  auto c = a + bv;

  EXPECT_EQ(
    c, (gt::gtensor_device<double, 2>{{111., 113., 115.}, {121., 123., 125.}}));
}
