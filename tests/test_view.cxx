
#include <gtest/gtest.h>

#include <gtensor/gtensor.h>

using namespace gt::placeholders;

TEST(view, slice_all)
{
  gt::gtensor<double, 2> a = {{11., 12., 13.}, {21., 22., 23.}};

  auto b = gt::view<2>(a, {gt::all(), gt::all()});
  EXPECT_EQ(b, a);
  auto b2 = a.view(_all, _all);
  EXPECT_EQ(b2, a);
}

TEST(view, slice_value)
{
  gt::gtensor<double, 2> a = {{11., 12., 13.}, {21., 22., 23.}};
  auto c = a.view(0, _all);
  EXPECT_EQ(c, (gt::gtensor<double, 1>{11., 21.}));

  auto d = a.view(_all, 1);
  EXPECT_EQ(d, (gt::gtensor<double, 1>{21., 22., 23.}));
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
}

TEST(gview, ownership)
{
  gt::gtensor<double, 2> a = {{11., 12., 13.}, {21., 22., 23.}};

  auto b = std::move(a).view(_all, _all);
  EXPECT_EQ(b, (gt::gtensor<double, 2>{{11., 12., 13.}, {21., 22., 23.}}));
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

  EXPECT_EQ(a + bv, (gt::gtensor_device<double, 2>{{111., 113., 115.},
                                                   {121., 123., 125.}}));
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

#ifdef GTENSOR_HAVE_DEVICE

TEST(gview, device_assign_all)
{
  gt::gtensor_device<double, 2> a{{11., 12., 13.}, {21., 22., 23.}};

  a.view(_all, _all) =
    gt::gtensor_device<double, 2>{{-11., -12., -13.}, {-21., -22., -23.}};

  EXPECT_EQ(a,
            (gt::gtensor<double, 2>{{-11., -12., -13.}, {-21., -22., -23.}}));
}

TEST(gview, device_assign_sub)
{
  gt::gtensor<double, 2> a{{11., 12., 13.}, {21., 22., 23.}};

  a.view(_s(1, 3), _s(0, 2)) =
    gt::gtensor<double, 2>{{-12., -13.}, {-22., -23.}};

  EXPECT_EQ(a, (gt::gtensor<double, 2>{{11., -12., -13.}, {21., -22., -23.}}));
}

#endif
