
#include <gtest/gtest.h>

#include <gtensor/gtensor.h>

TEST(view, slice_all)
{
  gt::gtensor<double, 2> a = {{11., 12., 13.}, {21., 22., 23.}};

  auto b = gt::view<2>(a, {gt::all(), gt::all()});
  EXPECT_EQ(b, a);
}

TEST(view, slice_value)
{
  gt::gtensor<double, 2> a = {{11., 12., 13.}, {21., 22., 23.}};
  auto c = gt::view<1>(a, {0, gt::all()});
  EXPECT_EQ(c, (gt::gtensor<double, 1>{11., 21.}));

  auto d = gt::view<1>(a, {gt::all(), 1});
  EXPECT_EQ(d, (gt::gtensor<double, 1>{21., 22., 23.}));
}

TEST(view, slice_range)
{
  gt::gtensor<double, 2> a = {{11., 12., 13.}, {21., 22., 23.}};

  auto b = gt::view<2>(a, {gt::slice(1, 3), gt::all()});
  EXPECT_EQ(b, (gt::gtensor<double, 2>{{12., 13.}, {22., 23.}}));
}

TEST(view, slice_missing)
{
  gt::gtensor<double, 2> a = {{11., 12., 13.}, {21., 22., 23.}};

  auto b = gt::view<2>(a, {gt::slice(1, 3)});
  EXPECT_EQ(b, (gt::gtensor<double, 2>{{12., 13.}, {22., 23.}}));
}

TEST(view, assign)
{
  gt::gtensor<double, 2> a = {{11., 12., 13.}, {21., 22., 23.}};

  auto b = gt::view<2>(a, {gt::slice(1, 3)});
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

  auto b = gt::view<2>(std::move(a), {gt::all(), gt::all()});
  EXPECT_EQ(b, (gt::gtensor<double, 2>{{11., 12., 13.}, {21., 22., 23.}}));
}

TEST(gview, assign_all)
{
  gt::gtensor<double, 2> a{{11., 12., 13.}, {21., 22., 23.}};
  auto b = gt::view<2>(a, {gt::all(), gt::all()});

  b = gt::gtensor<double, 2>{{-11., -12., -13.}, {-21., -22., -23.}};

  EXPECT_EQ(a,
            (gt::gtensor<double, 2>{{-11., -12., -13.}, {-21., -22., -23.}}));
}

TEST(gview, assign_sub)
{
  gt::gtensor<double, 2> a{{11., 12., 13.}, {21., 22., 23.}};
  auto b = gt::view<2>(a, {gt::slice(1, 3), gt::slice(0, 2)});

  b = gt::gtensor<double, 2>{{-12., -13.}, {-22., -23.}};

  EXPECT_EQ(a, (gt::gtensor<double, 2>{{11., -12., -13.}, {21., -22., -23.}}));
}

TEST(gview, newaxis)
{
  gt::gtensor<double, 2> a = {{11., 12., 13.}, {21., 22., 23.}};
  gt::gtensor<double, 1> b = {100., 101., 102.};

  auto bv = gt::view<2>(b, {gt::all(), gt::newaxis()});
  auto c = a + bv;

  EXPECT_EQ(
    c, (gt::gtensor_device<double, 2>{{111., 113., 115.}, {121., 123., 125.}}));
}

TEST(gview, expression_all)
{
  gt::gtensor<double, 2> a = {{11., 12., 13.}, {21., 22., 23.}};

  auto c = gt::view<2>(a + a, {gt::all(), gt::all()});

  EXPECT_EQ(c, (gt::gtensor<double, 2>{{22., 24., 26.}, {42., 44., 46.}}));
}

TEST(gview, expression_sub)
{
  gt::gtensor<double, 2> a = {{11., 12., 13.}, {21., 22., 23.}};

  auto c = gt::view<1>(a + a, {gt::slice(1, 3), 1});

  EXPECT_EQ(c, (gt::gtensor<double, 1>{44., 46.}));
}

TEST(gview, swapaxes_expression)
{
  gt::gtensor<double, 2> a = {{11., 12., 13.}, {21., 22., 23.}};

  auto c = gt::swapaxes(a + a, 0, 1);

  EXPECT_EQ(c, (gt::gtensor<double, 2>{{22., 42.}, {24., 44.}, {26., 46.}}));
}

#ifdef __CUDACC__

TEST(gview, device_assign_all)
{
  gt::gtensor_device<double, 2> a{{11., 12., 13.}, {21., 22., 23.}};
  auto b = gt::view<2>(a, {gt::all(), gt::all()});

  b = gt::gtensor_device<double, 2>{{-11., -12., -13.}, {-21., -22., -23.}};

  EXPECT_EQ(a,
            (gt::gtensor<double, 2>{{-11., -12., -13.}, {-21., -22., -23.}}));
}

TEST(gview, device_assign_sub)
{
  gt::gtensor<double, 2> a{{11., 12., 13.}, {21., 22., 23.}};
  auto b = gt::view<2>(a, {gt::slice(1, 3), gt::slice(0, 2)});

  b = gt::gtensor<double, 2>{{-12., -13.}, {-22., -23.}};

  EXPECT_EQ(a, (gt::gtensor<double, 2>{{11., -12., -13.}, {21., -22., -23.}}));
}

#endif
