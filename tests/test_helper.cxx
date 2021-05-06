
#include <gtest/gtest.h>

#include "gtensor/gtensor.h"
#include "gtensor/helper.h"

#include <tuple>
#include <type_traits>

#include "test_debug.h"

using namespace gt::placeholders;

TEST(helper, tuple_max)
{
  std::tuple<> t0;
  std::tuple<int> t1 = {5};
  std::tuple<int, int> t2 = {5, 10};
  std::tuple<int, int, int> t3 = {5, 10, 15};
  std::tuple<int, int, int> t3a = {25, 10, 15};

  auto id = [](auto& val) { return val; };
  EXPECT_EQ(gt::helper::max(id, t0), 0);
  EXPECT_EQ(gt::helper::max(id, t1), 5);
  EXPECT_EQ(gt::helper::max(id, t2), 10);
  EXPECT_EQ(gt::helper::max(id, t3), 15);
  EXPECT_EQ(gt::helper::max(id, t3a), 25);
}

TEST(helper, nd_initializer_list)
{
  using namespace gt::helper;

  nd_initializer_list_t<int, 1> nd1 = {1, 2, 3, 4, 5, 6};
  auto nd1shape = nd_initializer_list_shape<1>(nd1);
  EXPECT_EQ(nd1shape, gt::shape(6));

  nd_initializer_list_t<int, 2> nd2 = {{1, 2, 3}, {4, 5, 6}};
  auto nd2shape = nd_initializer_list_shape<2>(nd2);
  EXPECT_EQ(nd2shape, gt::shape(3, 2));

  nd_initializer_list_t<int, 3> nd3 = {{{
                                          1,
                                        },
                                        {
                                          2,
                                        },
                                        {
                                          3,
                                        }},
                                       {{
                                          4,
                                        },
                                        {
                                          5,
                                        },
                                        {
                                          6,
                                        }}};
  auto nd3shape = nd_initializer_list_shape<3>(nd3);
  EXPECT_EQ(nd3shape, gt::shape(1, 3, 2));
}

TEST(helper, to_expression_t)
{
  gt::gtensor<double, 1> a;
  auto a_view = a.view();
  using to_expr_view_type = gt::to_expression_t<decltype(a_view)>;

  GT_DEBUG_TYPE(a);
  GT_DEBUG_TYPE(a_view);
  GT_DEBUG_TYPE_NAME(to_expr_view_type);

  EXPECT_TRUE((std::is_same<decltype(a_view), to_expr_view_type>::value));
}

TEST(helper, const_view_to_expression_t)
{
  gt::gtensor<double, 1> a;
  const auto a_view = a.view();
  using to_expr_view_type = gt::to_expression_t<decltype(a_view)>;

  GT_DEBUG_TYPE(a);
  GT_DEBUG_TYPE(a_view);
  GT_DEBUG_TYPE_NAME(to_expr_view_type);

  EXPECT_TRUE((std::is_same<decltype(a_view), to_expr_view_type>::value));
}

TEST(helper, const_gtensor_to_expression_t)
{
  const gt::gtensor<double, 1> a;
  auto a_view = a.view();
  using to_expr_view_type = gt::to_expression_t<decltype(a_view)>;

  GT_DEBUG_TYPE(a);
  GT_DEBUG_TYPE(a_view);
  GT_DEBUG_TYPE_NAME(to_expr_view_type);

  EXPECT_TRUE((std::is_same<decltype(a_view), to_expr_view_type>::value));
}

TEST(helper, gtensor_to_kernel_t)
{
  gt::gtensor<double, 1> a;
  auto k_a = a.to_kernel();
  using to_kern_type = gt::to_kernel_t<decltype(a)>;

  GT_DEBUG_TYPE(a);
  GT_DEBUG_TYPE(k_a);
  GT_DEBUG_TYPE_NAME(to_kern_type);

  EXPECT_TRUE((std::is_same<decltype(k_a), to_kern_type>::value));
}

TEST(helper, const_gtensor_to_kernel_t)
{
  const gt::gtensor<double, 1> a;
  auto k_a = a.to_kernel();
  using to_kern_type = gt::to_kernel_t<decltype(a)>;

  GT_DEBUG_TYPE(a);
  GT_DEBUG_TYPE(k_a);
  GT_DEBUG_TYPE_NAME(to_kern_type);

  EXPECT_TRUE((std::is_same<decltype(k_a), to_kern_type>::value));
}

TEST(helper, gview_to_kernel_t)
{
  gt::gtensor<double, 1> a;
  auto a_view = a.view();
  auto k_view = a_view.to_kernel();
  using to_kern_view_type = gt::to_kernel_t<decltype(a_view)>;

  GT_DEBUG_TYPE(a);
  GT_DEBUG_TYPE(a_view);
  GT_DEBUG_TYPE(k_view);
  GT_DEBUG_TYPE_NAME(to_kern_view_type);

  EXPECT_TRUE((std::is_same<decltype(k_view), to_kern_view_type>::value));
}

TEST(helper, gview_const_to_kernel_t)
{
  gt::gtensor<double, 1> a;
  const auto a_view = a.view();
  auto k_view = a_view.to_kernel();
  using to_kern_view_type = gt::to_kernel_t<decltype(a_view)>;

  GT_DEBUG_TYPE(a);
  GT_DEBUG_TYPE(a_view);
  GT_DEBUG_TYPE(k_view);
  GT_DEBUG_TYPE_NAME(to_kern_view_type);

  EXPECT_TRUE((std::is_same<decltype(k_view), to_kern_view_type>::value));
}

TEST(helper, gview_gtensor_const_to_kernel_t)
{
  const gt::gtensor<double, 1> a;
  auto a_view = a.view();
  auto k_view = a_view.to_kernel();
  using to_kern_view_type = gt::to_kernel_t<decltype(a_view)>;

  GT_DEBUG_TYPE(a);
  GT_DEBUG_TYPE(a_view);
  GT_DEBUG_TYPE(k_view);
  GT_DEBUG_TYPE_NAME(to_kern_view_type);

  EXPECT_TRUE((std::is_same<decltype(k_view), to_kern_view_type>::value));
}

TEST(helper, const_gfunction_to_expression_t)
{
  gt::gtensor<double, 1> a;
  auto a_view = a.view();
  const auto gfn = 2. * a_view;
  using to_expr_fn_type = gt::to_expression_t<decltype(gfn)>;

  GT_DEBUG_TYPE(a);
  GT_DEBUG_TYPE(a_view);
  GT_DEBUG_TYPE(gfn);
  GT_DEBUG_TYPE_NAME(to_expr_fn_type);

  EXPECT_TRUE((std::is_same<decltype(gfn), to_expr_fn_type>::value));
}

TEST(helper, const_gfunction_to_kernel_t)
{
  gt::gtensor<double, 1> a;
  auto a_view = a.view();
  const auto gfn = 2. * a_view;
  auto k_gfn = gfn.to_kernel();
  using to_kern_fn_type = gt::to_kernel_t<decltype(gfn)>;

  GT_DEBUG_TYPE(a);
  GT_DEBUG_TYPE(a_view);
  GT_DEBUG_TYPE(gfn);
  GT_DEBUG_TYPE(k_gfn);
  GT_DEBUG_TYPE_NAME(to_kern_fn_type);

  EXPECT_TRUE((std::is_same<decltype(k_gfn), to_kern_fn_type>::value));
}

TEST(helper, const_gview_owner_to_kernel_t)
{
  const auto a_view = gt::view(gt::gtensor<double, 1>{1., 2., 3.}, _all);
  auto k_view = a_view.to_kernel();
  using to_kern_view_type = gt::to_kernel_t<decltype(a_view)>;

  GT_DEBUG_TYPE(a_view);
  GT_DEBUG_TYPE(k_view);
  GT_DEBUG_TYPE_NAME(to_kern_view_type);

  EXPECT_TRUE((std::is_same<decltype(k_view), to_kern_view_type>::value));
}

TEST(helper, const_gfunction_gview_owner_to_kernel_t)
{
  const auto gfn = 2. * gt::view(gt::gtensor<double, 1>{1., 2., 3.}, _all);
  auto k_gfn = gfn.to_kernel();
  using to_kern_fn_type = gt::to_kernel_t<decltype(gfn)>;

  GT_DEBUG_TYPE(gfn);
  GT_DEBUG_TYPE(k_gfn);
  GT_DEBUG_TYPE_NAME(to_kern_fn_type);

  EXPECT_TRUE((std::is_same<decltype(k_gfn), to_kern_fn_type>::value));
}

TEST(helper, has_data_method)
{
  gt::gtensor<double, 1> a(gt::shape(10));
  auto aspan = a.to_kernel();
  auto aview = a.view(gt::slice(3, 6));

  EXPECT_TRUE(gt::has_data_method_v<decltype(a)>);
  EXPECT_TRUE(gt::has_data_method<decltype(a)>::value);

  EXPECT_TRUE(gt::has_data_method_v<decltype(aspan)>);
  EXPECT_TRUE(gt::has_data_method<decltype(aspan)>::value);

  EXPECT_FALSE(gt::has_data_method_v<decltype(aview)>);
  EXPECT_FALSE(gt::has_data_method<decltype(aview)>::value);
}

TEST(helper, has_size_method)
{
  gt::gtensor<double, 1> a(gt::shape(10));
  auto aspan = a.to_kernel();
  auto aview = a.view(gt::slice(3, 6));

  EXPECT_TRUE(gt::has_size_method_v<decltype(a)>);
  EXPECT_TRUE(gt::has_size_method<decltype(a)>::value);

  EXPECT_TRUE(gt::has_size_method_v<decltype(aspan)>);
  EXPECT_TRUE(gt::has_size_method<decltype(aspan)>::value);

  EXPECT_TRUE(gt::has_size_method_v<decltype(aview)>);
  EXPECT_TRUE(gt::has_size_method<decltype(aview)>::value);
}

TEST(helper, has_container_methods)
{
  gt::gtensor<double, 1> a(gt::shape(10));
  auto aspan = a.to_kernel();
  auto aview = a.view(gt::slice(3, 6));

  EXPECT_TRUE(gt::has_container_methods_v<decltype(a)>);
  EXPECT_TRUE(gt::has_container_methods<decltype(a)>::value);

  EXPECT_TRUE(gt::has_container_methods_v<decltype(aspan)>);
  EXPECT_TRUE(gt::has_container_methods<decltype(aspan)>::value);

  EXPECT_FALSE(gt::has_container_methods_v<decltype(aview)>);
  EXPECT_FALSE(gt::has_container_methods<decltype(aview)>::value);
}

TEST(helper, is_allowed_element_type_conversion)
{
  static_assert(
    gt::is_allowed_element_type_conversion<double, const double>::value,
    "convert to const");
  static_assert(
    !gt::is_allowed_element_type_conversion<const double, double>::value,
    "convert from const");
  static_assert(
    gt::is_allowed_element_type_conversion<const double, const double>::value,
    "convert same");
}