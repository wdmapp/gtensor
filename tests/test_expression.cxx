
#include <gtest/gtest.h>

#include "gtensor/complex.h"
#include "gtensor/gtensor.h"

#ifdef __CUDACC__
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
#ifdef __CUDACC__
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

  auto e2 = 10. + t2;
  EXPECT_EQ(e2, (gt::gtensor<double, 1>{13., 14.}));

  auto e3 = t1 + 10.;
  EXPECT_EQ(e3, (gt::gtensor<double, 1>{11., 12.}));

  // rvalue
  auto e4 = gt::gtensor<double, 1>{1., 2.} + 10.;
  EXPECT_EQ(e4, (gt::gtensor<double, 1>{11., 12.}));
}

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
