#include <gtest/gtest.h>

#include "gtensor/complex.h"
#include "gtensor/gtensor.h"
#include "gtensor/reductions.h"

#include "test_debug.h"

#ifdef GTENSOR_HAVE_DEVICE
using space = gt::space::device;
#else
using space = gt::space::host;
#endif

template <typename E>
struct MemberTest
{
  using type = E;

  MemberTest(E&& e) : e_(e) {}

  E e_;
};

template <typename E>
MemberTest<E> make_MemberTest(E&& e)
{
  return MemberTest<E>(std::forward<E>(e));
}

TEST(expression, member_t)
{
  using T = double;
  using E = gt::gtensor<double, 2>;

  gt::assert_is_same<gt::to_expression_t<E>, E>();
  gt::assert_is_same<gt::to_expression_t<E&>, E&>();
  gt::assert_is_same<gt::to_expression_t<const E&>, const E&>();

  gt::assert_is_same<gt::to_expression_t<T>, gt::gscalar<T>>();
  gt::assert_is_same<gt::to_expression_t<T&>, gt::gscalar<T&>>();
  gt::assert_is_same<gt::to_expression_t<const T&>, gt::gscalar<const T&>>();
}

TEST(expression, space_t)
{
  using namespace gt::space;
  gt::assert_is_same<gt::space_t<host, host>, host>();
#ifdef GTENSOR_HAVE_DEVICE
  gt::assert_is_same<gt::space_t<device, device>, device>();
  gt::assert_is_same<gt::space_t<gt::space::any, device>, device>();
#endif
  gt::assert_is_same<gt::space_t<host, gt::space::any>, host>();
  gt::assert_is_same<gt::space_t<gt::space::any, gt::space::any>,
                     gt::space::any>();
}

TEST(expression, forward)
{
  using E = gt::gtensor<double, 2>;

  E e({2, 3});
  auto mt1 = make_MemberTest(e);
  gt::assert_is_same<decltype(mt1)::type, E&>();

  const E e2({2, 3});
  auto mt2 = make_MemberTest(e2);
  gt::assert_is_same<decltype(mt2)::type, const E&>();

  auto mt3 = make_MemberTest(E({2, 3}));
  gt::assert_is_same<decltype(mt3)::type, E>();

  using T = double;

  T t;
  auto mt4 = make_MemberTest(t);
  gt::assert_is_same<decltype(mt4)::type, T&>();

  const T t2 = 2.;
  auto mt5 = make_MemberTest(t2);
  gt::assert_is_same<decltype(mt5)::type, const T&>();

  auto mt6 = make_MemberTest(T());
  gt::assert_is_same<decltype(mt6)::type, T>();
}

TEST(expression, gfunction)
{
  gt::gtensor<double, 1> t1({1., 2.});
  gt::gtensor<double, 1> t2({3., 4.});

  auto e = t1 + t2;
  EXPECT_EQ(e, (gt::gtensor<double, 1>{4., 6.}));

  EXPECT_EQ(e.dimension(), 1);
  EXPECT_EQ(e.shape(), gt::shape(2));
  EXPECT_EQ(e.size(), 2);

  auto e2 = 10. + t2;
  EXPECT_EQ(e2, (gt::gtensor<double, 1>{13., 14.}));

  auto e3 = t1 + 10.;
  EXPECT_EQ(e3, (gt::gtensor<double, 1>{11., 12.}));

  // rvalue
  auto e4 = gt::gtensor<double, 1>{1., 2.} + 10.;
  EXPECT_EQ(e4, (gt::gtensor<double, 1>{11., 12.}));
}

TEST(expression, gfunction_unary)
{
  gt::gtensor<double, 1> t({1., 2.});

  auto e = -t;
  EXPECT_EQ(e.shape(), gt::shape(2));
  EXPECT_EQ(e.size(), 2);
  EXPECT_EQ(e, (gt::gtensor<double, 1>{-1., -2.}));
}

TEST(expression, gfunction_to_kernel)
{
  gt::gtensor<double, 1> t1({1., 2.});
  gt::gtensor<double, 1> t2({3., 4.});

  auto e = t1 + t2;
  auto k_e = e.to_kernel();
  EXPECT_EQ(k_e, (gt::gtensor<double, 1>{4., 6.}));

  EXPECT_EQ(k_e.dimension(), 1);
}

TEST(expression, gfunction_to_kernel_const)
{
  gt::gtensor<double, 1> t1({1., 2.});
  const gt::gtensor<double, 1> t2({3., 4.});

  auto e = t1 + t2;
  auto k_e = e.to_kernel();
  EXPECT_EQ(k_e, (gt::gtensor<double, 1>{4., 6.}));

  EXPECT_EQ(k_e.dimension(), 1);
}

TEST(expression, gfunction_to_kernel_const_view)
{
  gt::gtensor<double, 1> t1({1., 2.});
  const gt::gtensor<double, 1> t2({3., 4.});

  auto t2_view = t2.view(gt::all);

  auto e = t1 + t2_view;
  auto k_e = e.to_kernel();
  EXPECT_EQ(k_e, (gt::gtensor<double, 1>{4., 6.}));

  EXPECT_EQ(k_e.dimension(), 1);
}

TEST(expression, gfunction_to_kernel_const_view_assign)
{
  gt::gtensor<double, 1> t1({1., 2.});
  const gt::gtensor<double, 1> t2({3., 4.});
  gt::gtensor<double, 1> result(t1.shape());

  auto t2_view = t2.view(gt::all);

  auto e = t1 + t2_view;
  result = gt::eval(e);
  EXPECT_EQ(result, (gt::gtensor<double, 1>{4., 6.}));
}

TEST(expression, gscalar)
{
  gt::gtensor<double, 1> t1({1., 2.});

  auto e1 = 2. * t1;
  EXPECT_EQ(e1, (gt::gtensor<double, 1>{2., 4.}));

  double n = 2.;
  auto e2 = n * t1;
  EXPECT_EQ(e2, (gt::gtensor<double, 1>{2., 4.}));

  const double& rn = 2.;
  auto e3 = rn * t1;
  EXPECT_EQ(e3, (gt::gtensor<double, 1>{2., 4.}));
}

TEST(expression, gscalar_lambda)
{
  gt::gtensor<double, 1> t1({1., 2.});

  auto scale = [](const double s, const gt::gtensor<double, 1>& x) {
    return s * x;
  };

  auto e1 = scale(2., t1);
  EXPECT_EQ(e1, (gt::gtensor<double, 1>{2., 4.}));
}

TEST(expression, abs)
{
  gt::gtensor<double, 1> t1({1., -2.});
  auto e1 = gt::abs(t1);
  EXPECT_EQ(e1, (gt::gtensor<double, 1>{1., 2.}));
}

TEST(expression, abs_complex)
{
  using T = std::complex<double>;
  gt::gtensor<T, 1> t1({T(3., 4.), -2.});
  auto e1 = gt::abs(t1);
  EXPECT_EQ(e1, (gt::gtensor<double, 1>{5., 2.}));
}

TEST(expression, sin)
{
  gt::gtensor<double, 1> t({0., M_PI / 2., M_PI, 3 * M_PI / 2.});
  gt::gtensor<double, 1> ref({0., 1., 0., -1.});

  EXPECT_LT(gt::norm_linf(gt::sin(t) - ref), 1e-14);
}

TEST(expression, cos)
{
  gt::gtensor<double, 1> t({0., M_PI / 2., M_PI, 3 * M_PI / 2.});
  gt::gtensor<double, 1> ref({1., 0., -1., 0.});

  EXPECT_LT(gt::norm_linf(gt::cos(t) - ref), 1e-14);
}

TEST(expression, exp)
{
  gt::gtensor<double, 1> t({0., 1.});
  gt::gtensor<double, 1> ref({1., std::exp(1.)});

  EXPECT_LT(gt::norm_linf(gt::exp(t) - ref), 1e-14);
}

TEST(expression, exp_complex)
{
  using namespace std::complex_literals;
  gt::complex<double> I(0., 1.);

  gt::gtensor<double, 1> t({0., M_PI / 2., M_PI, 3 * M_PI / 2.});

  gt::gtensor<gt::complex<double>, 1> ref(t.shape());
  ref(0) = 1.;
  ref(1) = 1.i;
  ref(2) = -1.;
  ref(3) = -1.i;

  EXPECT_LT(gt::norm_linf(gt::exp(I * t) - ref), 1e-14);
}

// Note: not currently working on Intel SYCL host backend on github
// CI, but does work locally on GPU backend
#if defined(GTENSOR_HAVE_DEVICE) && !defined(GTENSOR_DEVICE_SYCL_HOST)
TEST(expression, device_exp_complex)
{
  using namespace std::complex_literals;
  gt::complex<double> I(0., 1.);

  gt::gtensor_device<double, 1> t({0., M_PI / 2., M_PI, 3 * M_PI / 2.});

  gt::gtensor_device<gt::complex<double>, 1> ref(gt::shape(4));
  gt::gtensor<gt::complex<double>, 1> h_ref(gt::shape(4));
  h_ref(0) = 1.;
  h_ref(1) = 1.i;
  h_ref(2) = -1.;
  h_ref(3) = -1.i;
  copy(h_ref, ref);

  EXPECT_LT(gt::norm_linf(gt::exp(I * t) - ref), 1e-14);
}
#endif

TEST(shape, broadcast_same)
{
  auto a = gt::shape(2, 3, 4);
  auto b = gt::shape(2, 3, 4);
  gt::broadcast_shape(a, b);
  EXPECT_EQ(a, gt::shape(2, 3, 4));
}

TEST(shape, broadcast_first)
{
  using S3 = gt::sarray<int, 3>;

  S3 a = {2, 1, 4};
  S3 b = {2, 3, 4};
  gt::broadcast_shape(a, b);
  EXPECT_EQ(a, (S3{2, 3, 4}));
}

TEST(shape, broadcast_second)
{
  using S3 = gt::sarray<int, 3>;

  S3 a = {2, 3, 4};
  S3 b = {2, 1, 4};
  gt::broadcast_shape(a, b);
  EXPECT_EQ(a, (S3{2, 3, 4}));
}

TEST(expression, shape_same)
{
  using S3 = gt::sarray<int, 3>;

  gt::gtensor<double, 3> t1({2, 3, 4});
  gt::gtensor<double, 3> t2({2, 3, 4});

  EXPECT_EQ(t1.shape(), (S3{2, 3, 4}));
  EXPECT_EQ(t2.shape(), (S3{2, 3, 4}));

  auto e1 = -t1;
  auto shape1 = e1.shape();
  EXPECT_EQ(e1.shape(), (S3{2, 3, 4}));

  auto e2 = t1 + t2;
  EXPECT_EQ(e2.shape(), (S3{2, 3, 4}));
}

TEST(expression, shape_first)
{
  using S3 = gt::sarray<int, 3>;

  gt::gtensor<double, 3> t1({2, 3, 4});
  gt::gtensor<double, 2> t2({3, 4});

  auto e = t1 + t2;
  EXPECT_EQ(e.shape(), (S3{2, 3, 4}));
}

TEST(expression, shape_second)
{
  using S3 = gt::sarray<int, 3>;

  gt::gtensor<double, 2> t1({3, 4});
  gt::gtensor<double, 3> t2({2, 3, 4});

  auto e = t1 + t2;
  EXPECT_EQ(e.shape(), (S3{2, 3, 4}));
}

TEST(expression, typestr)
{
  gt::gtensor<double, 2> a = {{1, 2}, {3, 4}};
  // d2 -> d for owning data, 2 for 2d
  // <double> for element type
  // {2, 2} is shape
  // {1, 2} is strides
  EXPECT_EQ(a.typestr(), "d2<double>{2, 2}{1, 2}");
  std::cout << "a     typestr " << a.typestr() << std::endl;

  auto aview = a.view(gt::all, 0);
  // v1 is for 1d view
  // (...) is the owning data object the view is over (above)
  // {2}{1} are the strides of the view
  EXPECT_EQ(aview.typestr(), "v1(d2<double>{2, 2}{1, 2}){2}{1}");
  std::cout << "aview typestr " << aview.typestr() << std::endl;

  auto adfn = 5.0 + a;
  // fn:(... OP ...) for standard binary ops
  EXPECT_EQ(adfn.typestr(), "fn:(5<double> + d2<double>{2, 2}{1, 2})");
  std::cout << "adfn  typestr " << adfn.typestr() << std::endl;

  auto avfn = 5 * aview;
  EXPECT_EQ(avfn.typestr(), "fn:(5<int> * v1(d2<double>{2, 2}{1, 2}){2}{1})");
  std::cout << "avfn  typestr " << avfn.typestr() << std::endl;

  auto asfn = 5 * a.to_kernel();
  EXPECT_EQ(asfn.typestr(), "fn:(5<int> * s2<double>{2, 2}{1, 2})");
  std::cout << "asfn  typestr " << asfn.typestr() << std::endl;

  // fn:NAME(...) for unary functions
  auto sina = gt::sin(a);
  EXPECT_EQ(sina.typestr(), "fn:sin(d2<double>{2, 2}{1, 2})");
  std::cout << "sina  typestr " << sina.typestr() << std::endl;
}

template <typename S>
void test_index_expression()
{
  gt::gtensor<double, 2, S> a = {{11., 12., 13.}, {21., 22., 23.}};
  gt::gtensor<double, 2, S> b = {{-11., -12., -13.}, {-21., -22., -23.}};
  gt::gtensor<double, 2> h_a(a.shape());

  auto linear_shape = gt::shape(a.size());
  auto strides = a.strides();

  auto k_a = a.to_kernel();
  auto k_b = b.to_kernel();

  gt::launch<1, S>(
    linear_shape, GT_LAMBDA(int i) {
      auto idx = unravel(i, strides);
      index_expression(k_a, idx) =
        index_expression(k_a, idx) + 2 * index_expression(k_b, idx);
    });

  gt::copy(a, h_a);
  EXPECT_EQ(h_a,
            (gt::gtensor<double, 2>{{-11., -12., -13.}, {-21., -22., -23.}}));

  auto aview = a.view(gt::all, 0);
  auto bview = a.view(gt::all, 1);
  auto h_aview = h_a.view(gt::all, 0);
  auto k_aview = aview.to_kernel();
  auto k_bview = bview.to_kernel();

  auto linear_shape2 = gt::shape(aview.size());
  auto strides2 = calc_strides(aview.shape());

  gt::launch<1, S>(
    linear_shape2, GT_LAMBDA(int i) {
      auto idx = unravel(i, strides2);
      index_expression(k_aview, idx) =
        index_expression(k_aview, idx) + index_expression(k_bview, idx);
    });

  gt::copy(a, h_a);
  EXPECT_EQ(h_aview, (gt::gtensor<double, 1>{-32., -34., -36.}));
}

TEST(expression, host_index_expression)
{
  test_index_expression<gt::space::host>();
}

#ifdef GTENSOR_HAVE_DEVICE

TEST(expression, device_eval)
{
  gt::gtensor_device<double, 2> a = {{11., 12., 13.}, {21., 22., 23.}};
  gt::gtensor_device<double, 2> b = {{-11., -12., -13.}, {-21., -22., -23.}};

  gt::gtensor<double, 2> h_a(a.shape());
  gt::gtensor<double, 2> h_c(a.shape());

  gt::copy(a, h_a);

  auto e1 = a + 2. * b; // -a
  GT_DEBUG_TYPE(e1);
  auto e2 = 4. * a + b; // 3a
  GT_DEBUG_TYPE(e2);
  auto e = (1. / 2.) * (e1 + e2);
  GT_DEBUG_TYPE(e);
  auto c = eval(e);
  GT_DEBUG_TYPE(c);

  gt::copy(c, h_c);

  EXPECT_EQ(h_c, h_a);
}

TEST(expression, device_index_expression)
{
  test_index_expression<gt::space::device>();
}

#endif
