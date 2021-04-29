
#include <gtest/gtest.h>

#include <gtensor/gtensor.h>

#include "test_debug.h"

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

TEST(gtensor, ctor_from_expr_unary)
{
  gt::gtensor<double, 2> a{{11., 12., 13.}, {21., 22., 23.}};
  gt::gtensor<double, 2> b = 2 * a + (-a);
  EXPECT_EQ(b, (gt::gtensor<double, 2>{{11., 12., 13.}, {21., 22., 23.}}));
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

TEST(gtensor, indexing_2d)
{
  gt::gtensor<double, 2> a{{11., 21., 31.}, {12., 22., 32.}};
  auto adata = a.data();
  EXPECT_EQ(a(0, 0), 11.);
  EXPECT_EQ(a(0, 0), adata[0]);
  EXPECT_EQ(a(1, 0), 21.);
  EXPECT_EQ(a(1, 0), adata[1]);
  EXPECT_EQ(a(2, 0), 31.);
  EXPECT_EQ(a(2, 0), adata[2]);
  EXPECT_EQ(a(0, 1), 12.);
  EXPECT_EQ(a(0, 1), adata[3]);
  EXPECT_EQ(a(1, 1), 22.);
  EXPECT_EQ(a(1, 1), adata[4]);
  EXPECT_EQ(a(2, 1), 32.);
  EXPECT_EQ(a(2, 1), adata[5]);
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

  // underlying storage copied
  auto adata = a.data();
  auto bdata = b.data();
  EXPECT_NE(adata, bdata);
}

TEST(gtensor, move_ctor)
{
  gt::gtensor<double, 1> a{11., 12., 13.};
  auto adata = a.data();

  auto b = std::move(a);
  auto bdata = b.data();

  EXPECT_EQ(b, (gt::gtensor<double, 1>{11., 12., 13.}));

  // verify no data was copied
  EXPECT_EQ(adata, bdata);
}

TEST(gtensor, copy_assign1)
{
  gt::gtensor<double, 1> a{11., 12., 13.};
  gt::gtensor<double, 1> b;
  b = a;

  EXPECT_EQ(b, a);
}

TEST(gtensor, copy_assign2)
{
  gt::gtensor<double, 2> a{{1., 2., 3.}, {2., 4., 6.}};
  gt::gtensor<double, 2> b;
  b = a;

  EXPECT_EQ(b.shape(), gt::shape(3, 2));

  for (int i = 0; i < a.shape(0); i++) {
    for (int j = 0; j < a.shape(1); j++) {
      EXPECT_EQ(a(i, j), static_cast<double>((i + 1) * (j + 1)));
    }
  }
  EXPECT_EQ(b, a);
}

TEST(gtensor, copy_assign3)
{
  gt::gtensor<double, 3> a{{{1., 2.}, {2., 4.}}, {{2., 4.}, {4., 8.}}};
  gt::gtensor<double, 3> b;
  b = a;

  EXPECT_EQ(b.shape(), gt::shape(2, 2, 2));

  for (int i = 0; i < a.shape(0); i++) {
    for (int j = 0; j < a.shape(1); j++) {
      for (int k = 0; k < a.shape(2); k++) {
        EXPECT_EQ(a(i, j, k), static_cast<double>((i + 1) * (j + 1) * (k + 1)));
      }
    }
  }
  EXPECT_EQ(b, a);
}

TEST(gtensor, move_assign)
{
  gt::gtensor<double, 1> a{11., 12., 13.};
  double* adata = a.data();
  gt::gtensor<double, 1> b;
  b = std::move(a);
  double* bdata = b.data();

  EXPECT_EQ(b, (gt::gtensor<double, 1>{11., 12., 13.}));

  // verify no data was copied
  EXPECT_EQ(adata, bdata);
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

TEST(gtensor, type_aliases)
{
  gt::gtensor<double, 1> h1(10);

  GT_DEBUG_TYPE_NAME(decltype(h1)::value_type);
  GT_DEBUG_TYPE_NAME(decltype(h1)::reference);
  GT_DEBUG_TYPE_NAME(decltype(h1)::const_reference);
  GT_DEBUG_TYPE_NAME(decltype(h1)::pointer);
  GT_DEBUG_TYPE_NAME(decltype(h1)::const_pointer);

  EXPECT_TRUE((std::is_same<decltype(h1)::value_type, double>::value));
  EXPECT_TRUE((std::is_same<decltype(h1)::reference, double&>::value));
  EXPECT_TRUE(
    (std::is_same<decltype(h1)::const_reference, const double&>::value));
  static_assert(std::is_same<decltype(h1)::pointer, double*>::value,
                "type mismatch");
  EXPECT_TRUE(
    (std::is_same<decltype(h1)::const_pointer, const double*>::value));
}

TEST(gtensor, index_by_shape)
{
  gt::gtensor<double, 2> a{{11., 21., 31.}, {12., 22., 32.}};

  EXPECT_EQ(a[gt::shape(0, 0)], 11.);
  EXPECT_EQ(a[gt::shape(1, 0)], 21.);
  EXPECT_EQ(a[gt::shape(2, 0)], 31.);
  EXPECT_EQ(a[gt::shape(0, 1)], 12.);
  EXPECT_EQ(a[gt::shape(1, 1)], 22.);
  EXPECT_EQ(a[gt::shape(2, 1)], 32.);
}

TEST(gtensor, is_expression_types)
{
  gt::gtensor<double, 1> a({3, 4});

  EXPECT_TRUE(gt::is_gcontainer<decltype(a)>::value);
  EXPECT_TRUE(gt::is_expression<decltype(a)>::value);
  EXPECT_FALSE(gt::is_gtensor_span<decltype(a)>::value);

  auto aview = a.view(gt::all);
  EXPECT_FALSE(gt::is_gcontainer<decltype(aview)>::value);
  EXPECT_TRUE(gt::is_expression<decltype(aview)>::value);
  EXPECT_FALSE(gt::is_gtensor_span<decltype(aview)>::value);

  auto aspan = a.to_kernel();
  EXPECT_FALSE(gt::is_gcontainer<decltype(aspan)>::value);
  EXPECT_TRUE(gt::is_expression<decltype(aspan)>::value);
  EXPECT_TRUE(gt::is_gtensor_span<decltype(aspan)>::value);
}

template <typename EC, gt::size_type N, typename T2,
          typename = std::enable_if_t<
            std::is_convertible<T2, typename EC::value_type>::value>>
void expect_all_eq(gt::gtensor_container<EC, N>& a, T2 value)
{
  using T = typename EC::value_type;
  auto aflat = gt::flatten(a);
  for (int i = 0; i < aflat.shape(0); i++) {
    EXPECT_EQ(aflat(i), T(value));
  }
}

template <typename T, typename S>
void test_fill_ctors()
{
  constexpr int N = 1;
  auto shape = gt::shape(4);
  gt::gtensor<T, N> h(shape);
  auto e = gt::gtensor<T, N, S>(shape);
  auto z = gt::gtensor<T, N, S>(shape, T(0));
  auto o = gt::gtensor<T, N, S>(shape, T(1));

  EXPECT_EQ(e.shape(), shape);

  EXPECT_EQ(z.shape(), shape);
  gt::copy(z, h);
  expect_all_eq(h, 0);

  EXPECT_EQ(o.shape(), shape);
  gt::copy(o, h);
  expect_all_eq(h, 1);
}

TEST(gtensor, fill_ctors)
{
  test_fill_ctors<int, gt::space::host>();
  test_fill_ctors<float, gt::space::host>();
  test_fill_ctors<double, gt::space::host>();
  test_fill_ctors<gt::complex<float>, gt::space::host>();
  test_fill_ctors<gt::complex<double>, gt::space::host>();
}

template <typename T, typename S>
void test_init_helpers()
{
  auto shape = gt::shape(4);
  // Note: dimension inferred from shape dimension
  auto h = gt::empty<T>(shape);
  auto e = gt::empty<T, S>(shape);
  auto z = gt::zeros<T, S>(shape);
  auto o = gt::full<T, S>(shape, 1);

  EXPECT_EQ(e.shape(), shape);

  EXPECT_EQ(z.shape(), shape);
  gt::copy(z, h);
  expect_all_eq(h, 0);

  EXPECT_EQ(o.shape(), shape);
  gt::copy(o, h);
  expect_all_eq(h, 1);
}

TEST(gtensor, init_helpers)
{
  test_init_helpers<int, gt::space::host>();
  test_init_helpers<float, gt::space::host>();
  test_init_helpers<double, gt::space::host>();
  test_init_helpers<gt::complex<float>, gt::space::host>();
  test_init_helpers<gt::complex<double>, gt::space::host>();
}

template <typename T, typename S>
void test_init_helpers_literal_shape()
{
  auto h1d = gt::empty<T>({4});
  auto h2d = gt::empty<T>({4, 5});

  auto e1d = gt::empty<T, S>({4});
  auto e2d = gt::empty<T, S>({4, 5});
  EXPECT_EQ(e1d.shape(), gt::shape(4));
  EXPECT_EQ(e2d.shape(), gt::shape(4, 5));

  auto z1d = gt::zeros<T, S>({4});
  auto z2d = gt::zeros<T, S>({4, 5});
  EXPECT_EQ(z1d.shape(), gt::shape(4));
  EXPECT_EQ(z2d.shape(), gt::shape(4, 5));
  gt::copy(z1d, h1d);
  expect_all_eq(h1d, 0);
  gt::copy(z2d, h2d);
  expect_all_eq(h2d, 0);

  auto o1d = gt::full<T, S>({4}, 1);
  auto o2d = gt::full<T, S>({4, 5}, 1);
  EXPECT_EQ(o1d.shape(), gt::shape(4));
  EXPECT_EQ(o2d.shape(), gt::shape(4, 5));
  gt::copy(o1d, h1d);
  expect_all_eq(h1d, 1);
  gt::copy(o2d, h2d);
  expect_all_eq(h2d, 1);
}

TEST(gtensor, init_helpers_literal_shape)
{
  test_init_helpers_literal_shape<int, gt::space::host>();
  test_init_helpers_literal_shape<float, gt::space::host>();
  test_init_helpers_literal_shape<double, gt::space::host>();
  test_init_helpers_literal_shape<gt::complex<float>, gt::space::host>();
  test_init_helpers_literal_shape<gt::complex<double>, gt::space::host>();
}

template <typename T, typename S>
void test_init_like_helpers()
{
  auto shape = gt::shape(4);
  auto h = gt::empty<T>(shape);
  auto d = gt::empty<T, S>(shape);
  auto el = gt::empty_like(d);
  auto zl = gt::zeros_like(d);
  auto ol = gt::full_like(d, 1);

  EXPECT_EQ(el.shape(), shape);

  gt::copy(zl, h);
  expect_all_eq(h, 0);

  gt::copy(ol, h);
  expect_all_eq(h, 1);
}

TEST(gtensor, init_like_helpers)
{
  test_init_like_helpers<int, gt::space::host>();
  test_init_like_helpers<float, gt::space::host>();
  test_init_like_helpers<double, gt::space::host>();
  test_init_like_helpers<gt::complex<float>, gt::space::host>();
  test_init_like_helpers<gt::complex<double>, gt::space::host>();
}

#if defined GTENSOR_HAVE_DEVICE

TEST(gtensor, device_assign_gtensor)
{
  gt::gtensor_device<double, 2> a{{11., 12., 13.}, {21., 22., 23.}};
  gt::gtensor_device<double, 2> b(a.shape());

  b = a;

  EXPECT_EQ(b,
            (gt::gtensor_device<double, 2>{{11., 12., 13.}, {21., 22., 23.}}));
}

TEST(gtensor, device_assign_gtensor2)
{
  gt::gtensor_device<double, 2> a{{1., 2., 3.}, {2., 4., 6.}};
  gt::gtensor_device<double, 2> b(a.shape());
  gt::gtensor<double, 2> h_b(a.shape());
  b = a;

  EXPECT_EQ(b.shape(), gt::shape(3, 2));

  gt::copy(b, h_b);

  for (int i = 0; i < b.shape(0); i++) {
    for (int j = 0; j < b.shape(1); j++) {
      EXPECT_EQ(h_b(i, j), static_cast<double>((i + 1) * (j + 1)));
    }
  }
  EXPECT_EQ(b, a);
}

TEST(gtensor, device_assign_gtensor3)
{
  gt::gtensor_device<double, 3> a{{{1., 2.}, {2., 4.}}, {{2., 4.}, {4., 8.}}};
  gt::gtensor_device<double, 3> b(a.shape());
  gt::gtensor<double, 3> h_b(a.shape());
  b = a;

  EXPECT_EQ(b.shape(), gt::shape(2, 2, 2));

  gt::copy(b, h_b);

  for (int i = 0; i < b.shape(0); i++) {
    for (int j = 0; j < b.shape(1); j++) {
      for (int k = 0; k < b.shape(2); k++) {
        EXPECT_EQ(h_b(i, j, k),
                  static_cast<double>((i + 1) * (j + 1) * (k + 1)));
      }
    }
  }
  EXPECT_EQ(b, a);
}

TEST(gtensor, device_assign_gtensor4)
{
  auto shape = gt::shape(64, 32, 16, 8);

  gt::gtensor<double, 4> h_a{shape};
  gt::gtensor<double, 4> h_b{shape};

  for (int i = 0; i < gt::calc_size(shape); i++) {
    h_a.data_access(i) = static_cast<double>(i);
  }
  gt::gtensor_device<double, 4> a{shape};
  gt::gtensor_device<double, 4> b{shape};

  // host to device
  gt::copy(h_a, a);

  // device to device
  b = a;

  // device to host
  gt::copy(b, h_b);

  EXPECT_EQ(h_b, h_a);
}

TEST(gtensor, device_assign_gtensor5)
{
  auto shape = gt::shape(64, 32, 16, 8, 4);

  gt::gtensor<double, 5> h_a{shape};
  gt::gtensor<double, 5> h_b{shape};

  for (int i = 0; i < gt::calc_size(shape); i++) {
    h_a.data_access(i) = static_cast<double>(i);
  }
  gt::gtensor_device<double, 5> a{shape};
  gt::gtensor_device<double, 5> b{shape};

  // host to device
  gt::copy(h_a, a);

  // device to device
  b = a;

  // device to host
  gt::copy(b, h_b);

  EXPECT_EQ(h_b, h_a);
}

TEST(gtensor, device_assign_gtensor6)
{
  auto shape = gt::shape(64, 32, 16, 8, 4, 2);

  gt::gtensor<double, 6> h_a{shape};
  gt::gtensor<double, 6> h_b{shape};

  for (int i = 0; i < gt::calc_size(shape); i++) {
    h_a.data_access(i) = static_cast<double>(i);
  }
  gt::gtensor_device<double, 6> a{shape};
  gt::gtensor_device<double, 6> b{shape};

  // host to device
  gt::copy(h_a, a);

  // device to device
  b = a;

  // device to host
  gt::copy(b, h_b);

  EXPECT_EQ(h_b, h_a);
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

TEST(gtensor, device_assign_expression_4d)
{
  auto a = gt::empty_device<double>(gt::shape(2, 3, 4, 5));
  auto h_a = gt::empty<double>(gt::shape(2, 3, 4, 5));
  for (int i = 0; i < h_a.size(); i++) {
    h_a.data()[i] = double(i);
  }
  gt::copy(h_a, a);

  auto b = a + a;

  auto h_b = gt::empty<double>(gt::shape(2, 3, 4, 5));
  gt::copy(gt::eval(b), h_b);
  for (int i = 0; i < h_b.size(); i++) {
    EXPECT_EQ(h_b.data()[i], 2. * i);
  }
}

TEST(gtensor, device_move_ctor)
{
  gt::gtensor<double, 1> h_a{11., 12., 13.};
  gt::gtensor<double, 1> h_b(h_a.shape());
  gt::gtensor_device<double, 1> a(h_a.shape());

  gt::copy(h_a, a);
  auto adata = a.data();

  auto b = std::move(a);
  auto bdata = b.data();

  // Note: explicit copy to avoid thrust implicit kernel for accessing
  // device vector from host. Aids in understanding thrust backend behavior.
  gt::copy(b, h_b);
  EXPECT_EQ(h_b, (gt::gtensor<double, 1>{11., 12., 13.}));

  // verify no data was copied
  EXPECT_EQ(adata, bdata);
}

TEST(gtensor, device_move_assign)
{
  gt::gtensor<double, 1> h_a{11., 12., 13.};
  gt::gtensor<double, 1> h_b(h_a.shape());
  gt::gtensor_device<double, 1> a(h_a.shape());
  gt::gtensor_device<double, 1> b(h_a.shape());

  gt::copy(h_a, a);
  auto adata = a.data();

  b = std::move(a);
  auto bdata = b.data();

  // Note: explicit copy to avoid thrust implicit kernel for accessing
  // device vector from host. Aids in understanding thrust backend behavior.
  gt::copy(b, h_b);
  EXPECT_EQ(h_b, (gt::gtensor<double, 1>{11., 12., 13.}));

  // verify no data was copied
  EXPECT_EQ(adata, bdata);
}

TEST(gtensor, synchronize)
{
  gt::gtensor_device<double, 2> a{{11., 12., 13.}, {21., 22., 23.}};
  gt::gtensor_device<double, 2> b(a.shape());
  gt::gtensor_device<double, 2> c(a.shape());

  b = a;

  // it's hard to force an async operation in pure gtensor, i.e. without using
  // vendor specific API. Not necessary here since the stream (cuda/HIP) or
  // queue will serialize multiple device copies, but at least we are
  // exercising the function call.
  gt::synchronize();

  c = b;

  EXPECT_EQ(c,
            (gt::gtensor_device<double, 2>{{11., 12., 13.}, {21., 22., 23.}}));
}

TEST(gtensor, device_fill_ctors)
{
  test_fill_ctors<int, gt::space::device>();
  test_fill_ctors<float, gt::space::device>();
  test_fill_ctors<double, gt::space::device>();
  test_fill_ctors<gt::complex<float>, gt::space::device>();
  test_fill_ctors<gt::complex<double>, gt::space::device>();
}

TEST(gtensor, device_init_helpers)
{
  test_init_helpers<int, gt::space::device>();
  test_init_helpers<float, gt::space::device>();
  test_init_helpers<double, gt::space::device>();
  test_init_helpers<gt::complex<float>, gt::space::device>();
  test_init_helpers<gt::complex<double>, gt::space::device>();
}

TEST(gtensor, device_init_like_helpers)
{
  test_init_like_helpers<int, gt::space::device>();
  test_init_like_helpers<float, gt::space::device>();
  test_init_like_helpers<double, gt::space::device>();
  test_init_like_helpers<gt::complex<float>, gt::space::device>();
  test_init_like_helpers<gt::complex<double>, gt::space::device>();
}

TEST(gtensor, device_init_helpers_literal_shape)
{
  test_init_helpers_literal_shape<int, gt::space::device>();
  test_init_helpers_literal_shape<float, gt::space::device>();
  test_init_helpers_literal_shape<double, gt::space::device>();
  test_init_helpers_literal_shape<gt::complex<float>, gt::space::device>();
  test_init_helpers_literal_shape<gt::complex<double>, gt::space::device>();
}

#endif // GTENSOR_HAVE_DEVICE

#if defined(GTENSOR_DEVICE_CUDA) || defined(GTENSOR_DEVICE_HIP)

__global__ void kernel_test(gt::gtensor_span_device<double, 1> d_a,
                            gt::gtensor_span_device<double, 1> d_b)
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
