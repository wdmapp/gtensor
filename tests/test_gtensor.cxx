
#include <gtest/gtest.h>

#include <gtensor/gtensor.h>

TEST(shape_type, ctor_args)
{
  auto shape = gt::shape(2, 3);
  EXPECT_EQ(shape, gt::shape_type<2>(2, 3));
}

TEST(gtensor, ctor_default)
{
  gt::gtensor<double, 2> a;
  EXPECT_EQ(a.shape(), gt::shape(0, 0));
}

TEST(gtensor, ctor_shape)
{
  gt::gtensor<double, 2> a({2, 3});
  EXPECT_EQ(a.shape(), gt::shape(2, 3));
}

TEST(gtensor, ctor_from_expr)
{
  gt::gtensor<double, 2> a{{11., 12., 13.}, {21., 22., 23.}};
  gt::gtensor<double, 2> b = a + a;
  EXPECT_EQ(b, (gt::gtensor<double, 2>{{22., 24., 26.}, {42., 44., 46.}}));
}

TEST(gtensor, ctor_init_1d)
{
  gt::gtensor<double, 1> b({3., 4., 5.});

  EXPECT_EQ(b.shape(), gt::shape(3));
  EXPECT_EQ(b, (gt::gtensor<double, 1>{3., 4., 5.}));
}

TEST(gtensor, ctor_init_2d)
{
  gt::gtensor<double, 2> a{{11., 12., 13.}, {21., 22., 23.}};
  EXPECT_EQ(a, (gt::gtensor<double, 2>{{11., 12., 13.}, {21., 22., 23.}}));
}

TEST(gtensor, op_equal)
{
  gt::gtensor<double, 2> a{{11., 12., 13.}, {21., 22., 23.}};
  gt::gtensor<double, 2> b{{11., 12., 13.}, {21., 22., 23.}};
  gt::gtensor<double, 2> c{{11., 12., 13.}, {21., 52., 23.}};
  gt::gtensor<double, 2> d{{11., 12}, {21., 22.}};
  gt::gtensor<double, 1> e{11., 12};

  EXPECT_EQ(a, b);
  EXPECT_NE(a, c);
  EXPECT_NE(a, d);
  EXPECT_NE(a, e);
}

TEST(gtensor, empty_like)
{
  gt::gtensor<double, 2> a{{11., 12., 13.}, {21., 22., 23.}};
  gt::gtensor<double, 2> b = gt::empty_like(a);

  EXPECT_EQ(a.shape(), b.shape());
}

TEST(gtensor, copy_ctor)
{
  gt::gtensor<double, 1> a{11., 12., 13.};
  auto b = a;

  EXPECT_EQ(b, a);
}

TEST(gtensor, move_ctor)
{
  gt::gtensor<double, 1> a{11., 12., 13.};
  auto b = std::move(a);

  EXPECT_EQ(b, (gt::gtensor<double, 1>{11., 12., 13.}));
}

TEST(gtensor, copy_assign)
{
  gt::gtensor<double, 1> a{11., 12., 13.};
  gt::gtensor<double, 1> b;
  b = a;

  EXPECT_EQ(b, a);
}

TEST(gtensor, move_assign)
{
  gt::gtensor<double, 1> a{11., 12., 13.};
  gt::gtensor<double, 1> b;
  b = std::move(a);

  EXPECT_EQ(b, (gt::gtensor<double, 1>{11., 12., 13.}));
}

TEST(gtensor, assign_expression_1d)
{
  gt::gtensor<double, 1> a{11., 12., 13.};
  auto b = gt::empty_like(a);

  b = a + a;
  EXPECT_EQ(b, (gt::gtensor<double, 1>{22., 24., 26.}));
}

TEST(gtensor, assign_expression_2d)
{
  gt::gtensor<double, 2> a{{11., 12., 13.}, {21., 22., 23.}};
  auto b = gt::empty_like(a);

  b = a + a;
  EXPECT_EQ(b, (gt::gtensor<double, 2>{{22., 24., 26.}, {42., 44., 46.}}));
}

TEST(gtensor, eval_lvalue)
{
  gt::gtensor<double, 2> a{{11., 12., 13.}, {21., 22., 23.}};
  auto&& b = gt::eval(a);
  gt::assert_is_same<gt::gtensor<double, 2>&, decltype(b)>();

  EXPECT_EQ(b, (gt::gtensor<double, 2>{{11., 12., 13.}, {21., 22., 23.}}));
}

TEST(gtensor, eval_rvalue)
{
  gt::gtensor<double, 2> a{{11., 12., 13.}, {21., 22., 23.}};
  auto&& b = gt::eval(std::move(a));
  gt::assert_is_same<decltype(b), gt::gtensor<double, 2>&&>();

  EXPECT_EQ(b, (gt::gtensor<double, 2>{{11., 12., 13.}, {21., 22., 23.}}));
}

TEST(gtensor, eval_expr)
{
  gt::gtensor<double, 2> a{{11., 12., 13.}, {21., 22., 23.}};
  auto&& b = gt::eval(a + a);
  // gt::assert_is_same<decltype(b), gt::gtensor<double, 2>&&>();

  EXPECT_EQ(b, (gt::gtensor<double, 2>{{22., 24., 26.}, {42., 44., 46.}}));
}

TEST(gtensor, assign_expression_2d_resize)
{
  gt::gtensor<double, 2> a{{11., 12., 13.}, {21., 22., 23.}};
  gt::gtensor<double, 2> b; // = a FIXME
  b = a + a;
  EXPECT_EQ(b, (gt::gtensor<double, 2>{{22., 24., 26.}, {42., 44., 46.}}));
}

#ifdef GTENSOR_HAVE_DEVICE

TEST(gtensor, device_assign_gtensor)
{
  gt::gtensor_device<double, 2> a{{11., 12., 13.}, {21., 22., 23.}};
  gt::gtensor_device<float, 2> b(a.shape());

  b = a;

  EXPECT_EQ(b,
            (gt::gtensor_device<double, 2>{{11., 12., 13.}, {21., 22., 23.}}));
}

TEST(gtensor, device_assign_to_view)
{
  gt::gtensor_device<double, 2> a{{11., 12., 13.}, {21., 22., 23.}};
  auto b = gt::view<2>(a, {gt::slice(1, 3), gt::slice(0, 2)});

  b = gt::gtensor_device<double, 2>{{-12., -13.}, {-22., -23.}};

  EXPECT_EQ(
    a, (gt::gtensor_device<double, 2>{{11., -12., -13.}, {21., -22., -23.}}));
}

TEST(gtensor, device_assign_expression)
{
  gt::gtensor_device<double, 2> a{{11., 12., 13.}, {21., 22., 23.}};
  auto b = gt::empty_like(a);

  b = a + a;

  EXPECT_EQ(b,
            (gt::gtensor_device<double, 2>{{22., 24., 26.}, {42., 44., 46.}}));
}

__global__ void kernel_test(gt::gtensor_view_device<double, 1> d_a,
                            gt::gtensor_view_device<double, 1> d_b)
{
  int i = threadIdx.x;
  if (i < d_b.shape(0)) {
    d_b(i) = d_a(i);
  }
}

template <typename F>
__global__ void kernel_test_lambda(gt::sarray<int, 1> shape, F f)
{
  int i = threadIdx.x;
  if (i < shape[0]) {
    f(i);
  }
}

TEST(gtensor_kernel, kernel_call)
{
  gt::gtensor<double, 1, gt::space::device> a{1., 2., 3};
  // FIXME, 1-d arrays ctor is ambiguous-ish
  gt::gtensor<double, 1, gt::space::device> b(gt::shape(3));

  gtLaunchKernel(kernel_test, 1, 3, 0, 0, a.to_kernel(), b.to_kernel());

  EXPECT_EQ(b, (gt::gtensor<double, 1>{1., 2., 3.}));
}

void lambda_test(const gt::gtensor_device<double, 1>& a,
                 gt::gtensor_device<double, 1>& b)
{
  auto k_a = a.to_kernel();
  auto k_b = b.to_kernel();

  auto lf = GT_LAMBDA(int i) mutable { k_b(i) = k_a(i); };

  gtLaunchKernel(kernel_test_lambda, 1, 3, 0, 0, b.shape(), lf);
}

TEST(gtensor_kernel, kernel_lambda_call)
{
  gt::gtensor<double, 1, gt::space::device> a{1., 2., 3};
  // FIXME, 1-d arrays ctor is ambiguous-ish
  gt::gtensor<double, 1, gt::space::device> b(gt::shape(3));

  lambda_test(a, b);

  EXPECT_EQ(b, (gt::gtensor<double, 1>{1., 2., 3.}));
}

#endif
