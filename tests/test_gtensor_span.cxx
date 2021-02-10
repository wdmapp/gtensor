
#include <gtest/gtest.h>

#include <gtensor/gtensor.h>

#include "test_debug.h"

TEST(gtensor_span, adapt_ctor_init_2d)
{
  gt::gtensor<double, 2> a({{11., 12., 13.}, {21., 22., 23.}});
  auto a_data = a.data();

  auto b = gt::adapt<2>(a_data, {2, 3});
  EXPECT_EQ(b, (gt::gtensor<double, 2>({{11., 12.}, {13., 21.}, {22., 23.}})));
}

// Note: the implicit conversion from gtensor won't work if this
// is templates on the gtensor_span types.
inline double first_element(const gt::gtensor_span<double, 2>& a)
{
  return *a.data();
}

TEST(gtensor_span, gtensor_implicit_conversion_ctor)
{
  gt::gtensor<double, 2> a({{11., 12., 13.}, {21., 22., 23.}});
  gt::gtensor_span<double, 2> aspan = a;
  gt::gtensor_span<const double, 2> aspan2 = a;

  const gt::gtensor<double, 2> b({{11., 12., 13.}, {21., 22., 23.}});
  // compile error
  // gt::gtensor_span<double, 2> bspan = b;
  gt::gtensor_span<const double, 2> bspan = b;

  GT_DEBUG_TYPE(aspan);
  GT_DEBUG_TYPE(aspan2);
  GT_DEBUG_TYPE(bspan);

  EXPECT_EQ(aspan, a);
  EXPECT_EQ(aspan2, a);
  EXPECT_EQ(bspan, b);

  double first = first_element(a);
  EXPECT_EQ(first, a(0, 0));
}

TEST(gtensor_span, adapt_vector)
{
  std::vector<double> v = {2., 3.};
  auto x = gt::adapt<1>(v.data(), {2});
  EXPECT_EQ(x, (gt::gtensor<double, 1>{2., 3.}));
  x = gt::gtensor<double, 1>{4., 5.};
  EXPECT_EQ(x, (gt::gtensor<double, 1>{4., 5.}));
}

TEST(gtensor_span, to_kernel_constness)
{
  gt::gtensor<double, 2> a({{11., 12., 13.}, {21., 22., 23.}});
  const gt::gtensor<double, 2> a_const({{11., 12., 13.}, {21., 22., 23.}});

  gt::gtensor_span<double, 2> a_view_type(a.data(), a.shape(), a.strides());
  gt::gtensor_span<const double, 2> a_const_view_type(
    a_const.data(), a_const.shape(), a_const.strides());

  auto a_view = a.to_kernel();
  auto a_const_view = a_const.to_kernel();

  using const_gtensor_span_type = gt::to_kernel_t<decltype(a_const)>;

  EXPECT_EQ(typeid(a_view).name(), typeid(a_view_type).name());
  EXPECT_EQ(typeid(a_const_view).name(), typeid(a_const_view_type).name());
  EXPECT_EQ(typeid(const_gtensor_span_type).name(),
            typeid(a_const_view_type).name());
}

TEST(gtensor_span, convert_const)
{
  constexpr int N = 10;
  gt::gtensor<float, 1> a(gt::shape(N));
  gt::gtensor_span<float, 1> a_view(a.data(), a.shape(), a.strides());
  gt::gtensor_span<float, 1> a_view_copy(a_view);
  const gt::gtensor_span<float, 1> a_view_const(a_view);
  const gt::gtensor_span<float, 1> a_view_const2 = a_view;

  for (int i = 0; i < N; i++) {
    a(i) = static_cast<float>(i);
  }

  // won't compile, different storage size so conversion ctor not defined
  // gt::gtensor_span<double, 1> double_view(a_view);

  EXPECT_EQ(a_view.data(), a_view_copy.data());
  EXPECT_EQ(&a_view(N - 1), &a_view_copy(N - 1));

  EXPECT_EQ(a_view.data(), a_view_const.data());
  EXPECT_EQ(&a_view(N - 1), &a_view_const(N - 1));

  EXPECT_EQ(a_view.data(), a_view_const2.data());
  EXPECT_EQ(&a_view(N - 1), &a_view_const2(N - 1));
}

TEST(gtensor_span, index_by_shape)
{
  gt::gtensor<double, 2> a{{11., 21., 31.}, {12., 22., 32.}};
  auto aspan = a.to_kernel();

  EXPECT_EQ(aspan[gt::shape(0, 0)], 11.);
  EXPECT_EQ(aspan[gt::shape(1, 0)], 21.);
  EXPECT_EQ(aspan[gt::shape(2, 0)], 31.);
  EXPECT_EQ(aspan[gt::shape(0, 1)], 12.);
  EXPECT_EQ(aspan[gt::shape(1, 1)], 22.);
  EXPECT_EQ(aspan[gt::shape(2, 1)], 32.);
}
