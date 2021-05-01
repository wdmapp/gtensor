#include <memory>

#include <type_traits>

#include <gtest/gtest.h>

#include <gtensor/gtensor.h>

#include "test_debug.h"

using namespace gt::placeholders;

// ======================================================================
// arange_0
//
// generates an integer sequence in the first (i) direction

template <typename T>
struct arange_0_generator
{
  GT_INLINE T operator()(int i, int j) const { return T(i); }
};

template <typename T>
auto arange_0(const gt::shape_type<2>& shape)
{
  return gt::generator<2, T>(shape, arange_0_generator<T>{});
}

// helper to test copy elision and moving behavior of returning a gtensor local
// from a function.
template <typename S>
auto get_linear_index_gtensor(gt::shape_type<2> shape)
{
  gt::gtensor<double, 2> h_a(shape);
#ifdef GTENSOR_HAVE_DEVICE
  gt::gtensor<double, 2, S> a(shape);
#endif

  int size = calc_size(shape);

  double* h_data = h_a.data();
  for (int i = 0; i < size; i++) {
    h_data[i] = i;
  }

#ifdef GTENSOR_HAVE_DEVICE
  gt::copy(h_a, a);
  return a;
#else
  return h_a;
#endif
}

TEST(view, slice_all)
{
  gt::gtensor<double, 2> a = {{11., 12., 13.}, {21., 22., 23.}};

  auto b = gt::view<2>(a, {gt::all, gt::all});
  EXPECT_EQ(b, a);
  auto b2 = a.view(_all, _all);
  EXPECT_EQ(b2, a);
}

TEST(view, slice_value)
{
  gt::gtensor<double, 2> a = {{11., 12., 13.}, {21., 22., 23.}};
  auto c = a.view(0, _all);
  EXPECT_EQ(c, (gt::gtensor<double, 1>{11., 21.}));

  // make sure that no data was copied
  EXPECT_EQ(std::addressof(c(0)), std::addressof(a(0, 0)));

  auto d = a.view(_all, 1);
  EXPECT_EQ(d, (gt::gtensor<double, 1>{21., 22., 23.}));

  // make sure that no data was copied
  EXPECT_EQ(std::addressof(d(0)), std::addressof(a(0, 1)));
}

TEST(view, slice_range)
{
  gt::gtensor<double, 2> a = {{11., 12., 13.}, {21., 22., 23.}};

  EXPECT_EQ(a.view(_s(1, 3), _all),
            (gt::gtensor<double, 2>{{12., 13.}, {22., 23.}}));

  gt::gtensor<double, 1> a1 = {0., 1., 2., 3., 4.};
  EXPECT_EQ(a1.view(_s(2, 4)), (gt::gtensor<double, 1>{2., 3.}));
  EXPECT_EQ(a1.view(_s(2, _)), (gt::gtensor<double, 1>{2., 3., 4.}));

  // FIXME? different behavior from numpy
  EXPECT_EQ(a1.view(_s(2, 0)), (gt::gtensor<double, 1>{2., 3., 4.}));

  EXPECT_EQ(a1.view(_s(0, 3)), (gt::gtensor<double, 1>{0., 1., 2.}));
  EXPECT_EQ(a1.view(_s(_, 3)), (gt::gtensor<double, 1>{0., 1., 2.}));
  EXPECT_EQ(a1.view(_s(_, _)), (gt::gtensor<double, 1>{0., 1., 2., 3., 4.}));

  // with step
  EXPECT_EQ(a1.view(_s(1, 4, _)), (gt::gtensor<double, 1>{1., 2., 3.}));
  EXPECT_EQ(a1.view(_s(1, 4, 1)), (gt::gtensor<double, 1>{1., 2., 3.}));
  EXPECT_EQ(a1.view(_s(1, 4, 2)), (gt::gtensor<double, 1>{1., 3.}));
  EXPECT_EQ(a1.view(_s(1, 2, 3)), (gt::gtensor<double, 1>{1.}));
  EXPECT_EQ(a1.view(_s(1, 3, 3)), (gt::gtensor<double, 1>{1.}));
  EXPECT_EQ(a1.view(_s(1, 4, 3)), (gt::gtensor<double, 1>{1.}));
  EXPECT_EQ(a1.view(_s(1, 5, 3)), (gt::gtensor<double, 1>{1., 4.}));
  EXPECT_EQ(a1.view(_s(1, _, 2)), (gt::gtensor<double, 1>{1., 3.}));
  EXPECT_EQ(a1.view(_s(_, 4, 2)), (gt::gtensor<double, 1>{0., 2.}));
  EXPECT_EQ(a1.view(_s(_, _, 2)), (gt::gtensor<double, 1>{0., 2., 4.}));

  // with negative step
  EXPECT_EQ(a1.view(_s(4, 1, -1)), (gt::gtensor<double, 1>{4., 3., 2.}));
  EXPECT_EQ(a1.view(_s(4, 1, -2)), (gt::gtensor<double, 1>{4., 2.}));
  EXPECT_EQ(a1.view(_s(3, 0, -2)), (gt::gtensor<double, 1>{3., 1.}));
  EXPECT_EQ(a1.view(_s(3, _, -2)), (gt::gtensor<double, 1>{3., 1.}));
  EXPECT_EQ(a1.view(_s(_, 1, -2)), (gt::gtensor<double, 1>{4., 2.}));
  EXPECT_EQ(a1.view(_s(_, _, -2)), (gt::gtensor<double, 1>{4., 2., 0.}));
}

TEST(view, slice_missing)
{
  gt::gtensor<double, 2> a = {{11., 12., 13.}, {21., 22., 23.}};

  EXPECT_EQ(a.view(_s(1, 3)), (gt::gtensor<double, 2>{{12., 13.}, {22., 23.}}));
}

TEST(view, assign)
{
  gt::gtensor<double, 2> a = {{11., 12., 13.}, {21., 22., 23.}};

  auto b = a.view(_s(1, 3));
  EXPECT_EQ(b, (gt::gtensor<double, 2>{{12., 13.}, {22., 23.}}));

  b(1, 1) = 99.;
  EXPECT_EQ(b, (gt::gtensor<double, 2>{{12., 13.}, {22., 99.}}));
  EXPECT_EQ(a, (gt::gtensor<double, 2>{{11., 12., 13.}, {21., 22., 99.}}));
}

TEST(view, reshape_view)
{
  gt::gtensor<double, 2> a = {{11., 12., 13.}, {21., 22., 23.}};

  auto b = gt::reshape<2>(a, {2, 3});
  EXPECT_EQ(b, (gt::gtensor<double, 2>{{11., 12.}, {13., 21.}, {22., 23.}}));

  // make sure that no data was copied
  EXPECT_EQ(std::addressof(a(0, 0)), std::addressof(b(0, 0)));
}

TEST(gview, ownership)
{
  gt::gtensor<double, 2> a = {{11., 12., 13.}, {21., 22., 23.}};

  auto b = std::move(a).view(_all, _all);
  EXPECT_EQ(b, (gt::gtensor<double, 2>{{11., 12., 13.}, {21., 22., 23.}}));
}

TEST(gview, copy_ctor)
{
  gt::gtensor<double, 2> a = {{11., 12., 13.}, {21., 22., 23.}};
  auto av = a.view(_s(1, 3), _all);
  auto av00_addr = std::addressof(av(0, 0));

  auto av2(av);
  EXPECT_EQ(av2, (gt::gtensor<double, 2>{{12., 13.}, {22., 23.}}));

  // make sure that underlying storage was not copied
  EXPECT_EQ(std::addressof(av2(0, 0)), av00_addr);
}

TEST(gview, fn_return_view)
{
  gt::gtensor<double, 2> a(gt::shape(2, 2));

  auto av = 2. * (get_linear_index_gtensor<gt::space::host>(gt::shape(3, 2))
                    .view(_s(1, _), _all));

  GT_DEBUG_TYPE(av);

  a = av;

  EXPECT_EQ(a, (gt::gtensor<double, 2>{{2., 4.}, {8., 10.}}));
}

TEST(gview, assign_all)
{
  gt::gtensor<double, 2> a{{11., 12., 13.}, {21., 22., 23.}};
  auto b = a.view(_all, _all);

  b = gt::gtensor<double, 2>{{-11., -12., -13.}, {-21., -22., -23.}};

  EXPECT_EQ(a,
            (gt::gtensor<double, 2>{{-11., -12., -13.}, {-21., -22., -23.}}));
}

TEST(gview, assign_sub)
{
  gt::gtensor<double, 2> a{{11., 12., 13.}, {21., 22., 23.}};
  auto b = a.view(_s(1, 3), _s(0, 2));

  b = gt::gtensor<double, 2>{{-12., -13.}, {-22., -23.}};

  EXPECT_EQ(a, (gt::gtensor<double, 2>{{11., -12., -13.}, {21., -22., -23.}}));
}

TEST(gview, newaxis)
{
  gt::gtensor<double, 2> a = {{11., 12., 13.}, {21., 22., 23.}};
  gt::gtensor<double, 1> b = {100., 101., 102.};

  auto bv = b.view(_all, _newaxis);

  EXPECT_EQ(a + bv,
            (gt::gtensor<double, 2>{{111., 113., 115.}, {121., 123., 125.}}));
}

TEST(gview, expression_all)
{
  gt::gtensor<double, 2> a = {{11., 12., 13.}, {21., 22., 23.}};

  auto c = gt::view<2>(a + a, {_all, _all});

  EXPECT_EQ(c, (gt::gtensor<double, 2>{{22., 24., 26.}, {42., 44., 46.}}));
}

TEST(gview, expression_sub)
{
  gt::gtensor<double, 2> a = {{11., 12., 13.}, {21., 22., 23.}};

  auto c = gt::view<1>(a + a, {_s(1, 3), 1});

  EXPECT_EQ(c, (gt::gtensor<double, 1>{44., 46.}));
}

TEST(gview, swapaxes_expression)
{
  gt::gtensor<double, 2> a = {{11., 12., 13.}, {21., 22., 23.}};

  auto c = gt::swapaxes(a + a, 0, 1);

  EXPECT_EQ(c, (gt::gtensor<double, 2>{{22., 42.}, {24., 44.}, {26., 46.}}));
}

TEST(gview, eval_view)
{
  gt::gtensor<double, 2> a = {{11., 12., 13.}, {21., 22., 23.}};
  gt::gtensor<double, 2> b = {{-11., -12., -13.}, {-21., -22., -23.}};

  auto av = a.view(_all, _all);
  auto bv = b.view(_all, _all);

  auto e1 = av + 2. * bv;         //   -a
  auto e2 = 4. * av + bv;         //  3*a
  auto e = (1. / 2.) * (e1 + e2); // 1/2 * (-a + 3a) = a
  auto c = gt::eval(e);

  EXPECT_EQ(c, a);
}

TEST(gview, rval_view)
{
  gt::gtensor<double, 2> a = {{11., 12., 13.}, {21., 22., 23.}};
  GT_DEBUG_VAR(a);
  GT_DEBUG_TYPE(a);

  auto multiply_view = [](const double m, const auto& x) {
    return m * x.view(_all, _all);
  };

  auto av = multiply_view(1., a);
  auto ar0 = -1. * arange_0<double>(a.shape());

  auto e1 = av + 2. * multiply_view(-1., a); //   -a
  GT_DEBUG_TYPE(e1);
  auto e2 = 4. * av + multiply_view(-1., a); //  3*a
  GT_DEBUG_TYPE(e2);
  auto e3 = 2. * ar0;
  GT_DEBUG_TYPE(e3);
  auto e = (1. / 2.) * (e1 + e2) + e3; // 1/2 * (-a + 3a) = a
  GT_DEBUG_TYPE(e);
  auto c = gt::eval(e);
  GT_DEBUG_TYPE(c);

  a = a + e3;

  EXPECT_EQ(c, a);
}

TEST(gview, gview_operator_parens)
{
  gt::gtensor<double, 2> a = {{11., 12., 13.}, {21., 22., 23.}};
  auto a_view = a.view(_s(1, 3), _all);
  using view_op_rtype = std::result_of_t<decltype(a_view)(int, int)>;

  GT_DEBUG_TYPE(a);
  GT_DEBUG_TYPE(a_view);
  GT_DEBUG_TYPE_NAME(view_op_rtype);

  EXPECT_TRUE((std::is_same<view_op_rtype, double&>::value));
}

TEST(gview, gview_const_gtensor_operator_parens)
{
  // non-const gview should NOT allow modification if wrapping const gtensor
  const gt::gtensor<double, 2> a = {{11., 12., 13.}, {21., 22., 23.}};
  auto a_view = a.view(_s(1, 3), _all);
  using view_op_rtype = std::result_of_t<decltype(a_view)(int, int)>;

  GT_DEBUG_TYPE(a);
  GT_DEBUG_TYPE(a_view);
  GT_DEBUG_TYPE_NAME(view_op_rtype);

  EXPECT_TRUE((std::is_same<view_op_rtype, const double&>::value));
}

TEST(gview, const_gview_operator_parens)
{
  // const gview should allow modification if wrapping non-const gtensor
  gt::gtensor<double, 2> a = {{11., 12., 13.}, {21., 22., 23.}};
  const auto a_view = a.view(_s(1, 3), _all);
  using view_op_rtype = std::result_of_t<decltype(a_view)(int, int)>;

  GT_DEBUG_TYPE(a);
  GT_DEBUG_TYPE(a_view);
  GT_DEBUG_TYPE_NAME(view_op_rtype);

  EXPECT_TRUE((std::is_same<view_op_rtype, double&>::value));
}

TEST(gview, gview_owner_operator_parens)
{
  // for an owning gview, the const-depth of the gview should be the
  // same as the depth of the owned expression. If it owns a gtensor,
  // this means it should allow modification if non-const view instance,
  // not allow if const.
  const auto const_view_gtensor_owner =
    gt::view(gt::gtensor<double, 1>{1., 2., 3.}, _all);
  using const_view_gtensor_owner_op_rtype =
    std::result_of_t<decltype(const_view_gtensor_owner)(int)>;

  GT_DEBUG_TYPE(const_view_gtensor_owner);
  GT_DEBUG_TYPE_NAME(const_view_gtensor_owner_op_rtype);

  EXPECT_TRUE(
    (std::is_same<const_view_gtensor_owner_op_rtype, const double&>::value));

  // non-const copy
  auto view_gtensor_owner = const_view_gtensor_owner;
  using view_gtensor_owner_op_rtype =
    std::result_of_t<decltype(view_gtensor_owner)(int)>;

  GT_DEBUG_TYPE(view_gtensor_owner);
  GT_DEBUG_TYPE_NAME(view_gtensor_owner_op_rtype);

  EXPECT_TRUE((std::is_same<view_gtensor_owner_op_rtype, double&>::value));
}

TEST(gview, dimension_arg_count)
{
  gt::gtensor<double, 2> a = {{11., 12., 13.}, {21., 22., 23.}};

  // less args then dimension - ok, assume others are _all
  auto a_view = a.view(_s(1, 3));
  EXPECT_EQ(a_view, (gt::gtensor<double, 2>{{12., 13.}, {22., 23.}}));

  // more args using _newaxis - ok
  auto a_newaxis = a.view(_s(1, 3), _all, _newaxis);
  EXPECT_EQ(a_newaxis, (gt::gtensor<double, 3>{{{12., 13.}, {22., 23.}}}));

  // too many args, not new axis - compile error
  // auto a_view_bad = a.view(_s(1, 3), _all, _all);
}

TEST(gview, index_by_shape)
{
  gt::gtensor<double, 2> a{{11., 21., 31.}, {12., 22., 32.}};
  auto aview = a.view(_all, _all);

  EXPECT_EQ(aview[gt::shape(0, 0)], 11.);
  EXPECT_EQ(aview[gt::shape(1, 0)], 21.);
  EXPECT_EQ(aview[gt::shape(2, 0)], 31.);
  EXPECT_EQ(aview[gt::shape(0, 1)], 12.);
  EXPECT_EQ(aview[gt::shape(1, 1)], 22.);
  EXPECT_EQ(aview[gt::shape(2, 1)], 32.);
}

TEST(gview, view_of_view)
{
  gt::gtensor<double, 2> a{{11., 21., 31.}, {12., 22., 32.}};

  auto aview = a.view(_s(1, 3), _all);
  EXPECT_EQ(aview, (gt::gtensor<double, 2>{{21., 31.}, {22., 32.}}));

  auto aview2 = gt::view(aview, _all, _s(1, 2));
  EXPECT_EQ(aview2, (gt::gtensor<double, 2>{{22., 32.}}));

  auto aview3 = gt::view(aview, _s(1, 2), _all);
  EXPECT_EQ(aview3, (gt::gtensor<double, 2>{{31.}, {32.}}));
}

TEST(gview, transpose_view)
{
  gt::gtensor<double, 2> a{{11., 21., 31.}, {12., 22., 32.}};
  auto aview = a.view(_s(1, 2), _all);
  EXPECT_EQ(aview.shape(), gt::shape(1, 2));
  EXPECT_EQ(aview, (gt::gtensor<double, 2>{{21.}, {22.}}));

  auto avtranspose = gt::transpose(aview, gt::shape(1, 0));
  EXPECT_EQ(avtranspose, (gt::gtensor<double, 2>{{21., 22.}}));
}

TEST(gview, reshape_view)
{
  gt::gtensor<double, 2> a{{11., 21., 31.}, {12., 22., 32.}};
  auto aview = a.view(_s(1, 2), _all);
  EXPECT_EQ(aview.shape(), gt::shape(1, 2));
  EXPECT_EQ(aview, (gt::gtensor<double, 2>{{21.}, {22.}}));

  auto aview2 = gt::reshape(aview, gt::shape(2, 1));
  EXPECT_EQ(aview2, (gt::gtensor<double, 2>{{21., 22.}}));
}

TEST(gview, reshape_reshape)
{
  gt::gtensor<double, 2> a{{11., 21., 31.}, {12., 22., 32.}};

  auto aview = gt::reshape(a, gt::shape(2, 3));
  EXPECT_EQ(aview,
            (gt::gtensor<double, 2>{{11., 21.}, {31., 12.}, {22., 32.}}));

  auto aview2 = gt::reshape(aview, gt::shape(6, 1));
  EXPECT_EQ(aview2, (gt::gtensor<double, 2>{{11., 21., 31., 12., 22., 32.}}));
}

TEST(gview, flatten_gtensor)
{
  gt::gtensor<double, 2> a{{11., 21., 31.}, {12., 22., 32.}};
  auto aflat = gt::flatten(a);

  EXPECT_EQ(aflat, (gt::gtensor<double, 1>{11., 21., 31., 12., 22., 32.}));
}

TEST(gview, flatten_view)
{
  gt::gtensor<double, 2> a{{11., 21., 31.}, {12., 22., 32.}};

  auto aview = a.view(_s(1, 3), _all);
  EXPECT_EQ(aview, (gt::gtensor<double, 2>{{21., 31.}, {22., 32.}}));

  auto aflat = gt::flatten(aview);
  EXPECT_EQ(aflat, (gt::gtensor<double, 1>{21., 31., 22., 32.}));
}

#ifdef GTENSOR_HAVE_DEVICE

TEST(gview, device_copy_ctor)
{
  gt::gtensor<double, 2> h_a = {{11., 12., 13.}, {21., 22., 23.}};
  gt::gtensor_device<double, 2> d_a(h_a.shape());

  gt::copy(h_a, d_a);

  auto d_av = d_a.view(_s(1, 3), _all);
  auto d_av00_addr = &d_av(0, 0);

  auto d_av2(d_av);

  // make sure that no data was copied
  EXPECT_EQ(&d_av2(0, 0), d_av00_addr);
}

// Note: this case reproduces a performance regression in GENE, where a
// thrust kernel is run to perform an internal copy. Fixed by adding
// a move constructor to gview class.
TEST(gview, device_fn_return_view)
{
  gt::gtensor<double, 2> h_a(gt::shape(2, 2));
  gt::gtensor_device<double, 2> a(gt::shape(2, 2));

  // construct an owning view from the gtensor returned by
  // get_linear_index_gtensor, and futher construct an expression containing
  // that view object. This results in the view object passing into a
  // gfunction object via universal references and std::forward calls.
  // Without a move constructor, this will result in a copy operation.
  auto av = 2. * (get_linear_index_gtensor<gt::space::device>(gt::shape(3, 2))
                    .view(_s(1, 3), _all));

  GT_DEBUG_TYPE(av);

  a = av;

  gt::copy(a, h_a);
  EXPECT_EQ(h_a, (gt::gtensor<double, 2>{{2., 4.}, {8., 10.}}));
}

TEST(gview, device_rval_view)
{
  gt::gtensor<double, 2> h_a = {{11., 12., 13.}, {21., 22., 23.}};
  gt::gtensor<double, 2> h_c(h_a.shape());
  gt::gtensor_device<double, 2> a(h_a.shape());

  gt::copy(h_a, a);

  auto multiply_view = [&](const double m, const auto& x) {
    return m * x.view(_all, _all);
  };

  auto av = multiply_view(1., a);

  auto ar0 = -1. * arange_0<double>(a.shape());

  auto e1 = av + 2. * multiply_view(-1., a); //   -a
  auto e2 = 4. * av + multiply_view(-1., a); //  3*a
  auto e3 = 2. * ar0;
  auto e = (1. / 2.) * (e1 + e2) + e3; // 1/2 * (-a + 3a) = a
  auto c = gt::eval(e);

  gt::copy(c, h_c);

  h_a = h_a + e3;

  EXPECT_EQ(h_c, h_a);
}

TEST(gview, device_assign_all)
{
  gt::gtensor_device<double, 2> a{{11., 12., 13.}, {21., 22., 23.}};
  gt::gtensor<double, 2> h_a(a.shape());

  a.view(_all, _all) =
    gt::gtensor_device<double, 2>{{-11., -12., -13.}, {-21., -22., -23.}};

  gt::copy(a, h_a);

  EXPECT_EQ(h_a,
            (gt::gtensor<double, 2>{{-11., -12., -13.}, {-21., -22., -23.}}));
}

TEST(gview, device_assign_sub)
{
  gt::gtensor_device<double, 2> a{{11., 12., 13.}, {21., 22., 23.}};
  gt::gtensor<double, 2> h_a(a.shape());

  a.view(_s(1, 3), _s(0, 2)) =
    gt::gtensor_device<double, 2>{{-12., -13.}, {-22., -23.}};

  gt::copy(a, h_a);

  EXPECT_EQ(h_a,
            (gt::gtensor<double, 2>{{11., -12., -13.}, {21., -22., -23.}}));
}

// Note: can be run with nvprof / nsys profile to see if thrust kernels
// are called unnecessarily
TEST(gview, device_eval_view)
{
  gt::gtensor_device<double, 2> a = {{11., 12., 13.}, {21., 22., 23.}};
  gt::gtensor_device<double, 2> b = {{-11., -12., -13.}, {-21., -22., -23.}};

  gt::gtensor<double, 2> h_a(a.shape());
  gt::gtensor<double, 2> h_c(a.shape());

  gt::copy(a, h_a);

  auto av = a.view(_all, _all);
  auto bv = b.view(_all, _all);

  auto e1 = av + 2. * bv; //  -a
  GT_DEBUG_TYPE(e1);
  auto e2 = 4. * av + bv; //  3a
  GT_DEBUG_TYPE(e2);
  auto e = (1. / 2.) * (e1 + e2); // 1/2 * (-a + 3a) = a
  GT_DEBUG_TYPE(e);
  auto c = eval(e);
  GT_DEBUG_TYPE(c);

  gt::copy(c, h_c);

  EXPECT_EQ(h_c, h_a);
}

#endif
