
#include <gtest/gtest.h>

#include <gtensor/gtensor.h>

TEST(gtensor, ctor_init_2d)
{
  gt::gtensor<double, 2> a({{11., 12., 13.}, {21., 22., 23.}});
  auto a_data = a.data();

  auto b = gt::adapt<2>(a_data, {2, 3});
  EXPECT_EQ(b, (gt::gtensor<double, 2>({{11., 12.}, {13., 21.}, {22., 23.}})));
}

TEST(gtensor_adaptor, vector)
{
  std::vector<double> v = {2., 3.};
  auto x = gt::adapt<1>(v.data(), {2});
  EXPECT_EQ(x, (gt::gtensor<double, 1>{2., 3.}));
  x = gt::gtensor<double, 1>{4., 5.};
  EXPECT_EQ(x, (gt::gtensor<double, 1>{4., 5.}));
}
