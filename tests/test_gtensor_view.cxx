
#include <gtest/gtest.h>

#include <gtensor/gtensor.h>

TEST(gtensor_view, adapt_ctor_init_2d)
{
  gt::gtensor<double, 2> a({{11., 12., 13.}, {21., 22., 23.}});
  auto a_data = a.data();

  auto b = gt::adapt<2>(a_data, {2, 3});
  EXPECT_EQ(b, (gt::gtensor<double, 2>({{11., 12.}, {13., 21.}, {22., 23.}})));
}

TEST(gtensor_view, adapt_vector)
{
  std::vector<double> v = {2., 3.};
  auto x = gt::adapt<1>(v.data(), {2});
  EXPECT_EQ(x, (gt::gtensor<double, 1>{2., 3.}));
  x = gt::gtensor<double, 1>{4., 5.};
  EXPECT_EQ(x, (gt::gtensor<double, 1>{4., 5.}));
}

TEST(gtensor_view, to_kernel_constness)
{
  gt::gtensor<double, 2> a({{11., 12., 13.}, {21., 22., 23.}});
  const gt::gtensor<double, 2> a_const({{11., 12., 13.}, {21., 22., 23.}});

  gt::gtensor_view<double, 2> a_view_type(a.data(), a.shape(), a.strides());
  gt::gtensor_view<const double, 2> a_const_view_type(
    a_const.data(), a_const.shape(), a_const.strides());

  auto a_view = a.to_kernel();
  auto a_const_view = a_const.to_kernel();

  using const_gtensor_view_type = gt::to_kernel_t<decltype(a_const)>;

  EXPECT_EQ(typeid(a_view).name(), typeid(a_view_type).name());
  EXPECT_EQ(typeid(a_const_view).name(), typeid(a_const_view_type).name());
  EXPECT_EQ(typeid(const_gtensor_view_type).name(),
            typeid(a_const_view_type).name());
}
