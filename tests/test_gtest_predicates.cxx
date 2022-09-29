#include <string>

#include <gtest/gtest.h>

#include "gtensor/gtensor.h"

#include "gtest_predicates.h"
#include "test_debug.h"

using namespace gt::placeholders;

TEST(gtest_predicates, fp)
{
  double a = 1.0;
  double b = 0.5;
  GT_EXPECT_EQ(a, 2 * b);

  float af = 1.0;
  float bf = 0.5;
  GT_EXPECT_EQ(af, 2 * bf);
}

TEST(gtest_predicates, complex)
{
  using D = gt::complex<double>;
  D a = D(-1.0, 0.0);
  D b = D(0.0, 1.0);
  GT_EXPECT_EQ(a, b * b);

  using F = gt::complex<float>;
  F af = F(-1.0, 0.0);
  F bf = F(0.0, 1.0);
  GT_EXPECT_EQ(a, b * b);
}

TEST(gtest_predicates, complex_fp)
{
  using D = gt::complex<double>;
  D im_d = D(0.0, -1.0);
  GT_EXPECT_EQ(im_d * im_d, -1.0);

  using F = gt::complex<float>;
  F im_f = F(0.0, -1.0);
  GT_EXPECT_EQ(im_f * im_f, -1.0);
}

TEST(gtest_predicates, gtensor_expr)
{
  gt::gtensor<double, 1> a{1, 2, 3};
  gt::gtensor<double, 1> b{0.5, 1, 1.5};
  gt::gtensor<double, 1> c{0.5, -1, 1.5};

  GT_EXPECT_NEAR(a, 2 * b);

  auto ar = gt::test::pred_near3("a", "c", "err", a, 2 * c, 0.0001);
  EXPECT_FALSE(ar);
  EXPECT_NE(std::string(ar.message()).find("at [1]"), std::string::npos);
}

TEST(gtest_predicates, gtensor_value)
{
  gt::gtensor<double, 1> a{1.001, 1.000001, 1.000000000001};

  GT_EXPECT_NEAR_MAXERR(a, 1.0, 0.01);

  auto ar = gt::test::pred_near3("a", "1", "err", a, 1.0, 0.0001);
  EXPECT_FALSE(ar);
  EXPECT_NE(std::string(ar.message()).find("at [0]"), std::string::npos);
}
