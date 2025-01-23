#include <gtest/gtest.h>

#include <gtensor/gtensor.h>

#include <gtensor/mxp.h>

#include <vector>

TEST(mxp, demo_axaxaxpy_implicit)
{
  const int n{2};
  const float x_init{1.f / 8.f / 1024.f / 1024.f};
  const float y_init{1.f};
  const float a{1.f / 3.f};

  EXPECT_NE(y_init, y_init + x_init);

  const std::vector<float> x(n, x_init);
  /* */ std::vector<float> y(n, y_init);

  const auto gt_x = gt::adapt<1>(x.data(), x.size());
  /* */ auto gt_y = gt::adapt<1>(y.data(), y.size());

  gt_y = gt_y + a * gt_x + a * gt_x + a * gt_x;

  EXPECT_EQ(y[0], y_init);
  EXPECT_EQ(y[1], y_init);

  const auto mxp_x = mxp::adapt<1, double>(x.data(), x.size());
  /* */ auto mxp_y = mxp::adapt<1, double>(y.data(), y.size());

  mxp_y = mxp_y + a * mxp_x + a * mxp_x + a * mxp_x;

  EXPECT_EQ(y[0], y_init + x_init);
  EXPECT_EQ(y[1], y_init + x_init);
}

TEST(mxp, demo_axaxaxpy_explicit)
{
  const int n{2};
  const float x_init{1.f / 8.f / 1024.f / 1024.f};
  const float y_init{1.f};
  const float a{1.f / 3.f};

  EXPECT_NE(y_init, y_init + x_init);

  const std::vector<float> x(n, x_init);
  /* */ std::vector<float> y(n, y_init);

  const auto gt_x = gt::adapt<1>(x.data(), x.size());
  /* */ auto gt_y = gt::adapt<1>(y.data(), y.size());

  gt::launch<1>(
    {n}, GT_LAMBDA(int j) {
      gt_y(j) = gt_y(j) + a * gt_x(j) + a * gt_x(j) + a * gt_x(j);
    });

  EXPECT_EQ(y[0], y_init);
  EXPECT_EQ(y[1], y_init);

  const auto mxp_x = mxp::adapt<1, double>(x.data(), x.size());
  /* */ auto mxp_y = mxp::adapt<1, double>(y.data(), y.size());

  gt::launch<1>(
    {n}, GT_LAMBDA(int j) {
      mxp_y(j) = mxp_y(j) + a * mxp_x(j) + a * mxp_x(j) + a * mxp_x(j);
    });

  EXPECT_EQ(y[0], y_init + x_init);
  EXPECT_EQ(y[1], y_init + x_init);
}
