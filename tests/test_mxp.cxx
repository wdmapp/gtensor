#include <gtest/gtest.h>

#include <gtensor/gtensor.h>

#include <gtensor/mxp.h>

#include <cmath>
#include <vector>

TEST(mxp, axaxaxpy_implicit)
{
  const int n{2};
  const float x_init{exp2f(-23)};
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

TEST(mxp, axaxaxpy_explicit)
{
  const int n{2};
  const float x_init{exp2f(-23)};
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

TEST(mxp, aXaXaXpY_2D_implicit)
{
  const int mn[2]{2, 3};
  const float x_init{exp2f(-23)};
  const float y_init{1.f};
  const float a{1.f / 3.f};

  EXPECT_NE(y_init, y_init + x_init);

  const std::vector<float> X(mn[0] * mn[1], x_init);
  /* */ std::vector<float> Y(mn[0] * mn[1], y_init);

  const auto gt_X = gt::adapt<2>(X.data(), mn);
  /* */ auto gt_Y = gt::adapt<2>(Y.data(), mn);

  gt_Y = gt_Y + a * gt_X + a * gt_X + a * gt_X;

  EXPECT_EQ(Y[0], y_init);
  EXPECT_EQ(Y[1], y_init);
  EXPECT_EQ(Y[2], y_init);
  EXPECT_EQ(Y[3], y_init);
  EXPECT_EQ(Y[4], y_init);
  EXPECT_EQ(Y[5], y_init);

  const auto mxp_X = mxp::adapt<2, double>(X.data(), mn);
  /* */ auto mxp_Y = mxp::adapt<2, double>(Y.data(), mn);

  mxp_Y = mxp_Y + a * mxp_X + a * mxp_X + a * mxp_X;

  EXPECT_EQ(Y[0], y_init + x_init);
  EXPECT_EQ(Y[1], y_init + x_init);
  EXPECT_EQ(Y[2], y_init + x_init);
  EXPECT_EQ(Y[3], y_init + x_init);
  EXPECT_EQ(Y[4], y_init + x_init);
  EXPECT_EQ(Y[5], y_init + x_init);
}

TEST(mxp, aXaXaXpY_2D_explicit)
{
  const int mn[2]{2, 3};
  const float x_init{exp2f(-23)};
  const float y_init{1.f};
  const float a{1.f / 3.f};

  EXPECT_NE(y_init, y_init + x_init);

  const std::vector<float> X(mn[0] * mn[1], x_init);
  /* */ std::vector<float> Y(mn[0] * mn[1], y_init);

  const auto gt_X = gt::adapt<2>(X.data(), mn);
  /* */ auto gt_Y = gt::adapt<2>(Y.data(), mn);

  gt::launch<2>(
    {mn[0], mn[1]}, GT_LAMBDA(int j, int k) {
      gt_Y(j, k) =
        gt_Y(j, k) + a * gt_X(j, k) + a * gt_X(j, k) + a * gt_X(j, k);
    });

  EXPECT_EQ(Y[0], y_init);
  EXPECT_EQ(Y[1], y_init);
  EXPECT_EQ(Y[2], y_init);
  EXPECT_EQ(Y[3], y_init);
  EXPECT_EQ(Y[4], y_init);
  EXPECT_EQ(Y[5], y_init);

  const auto mxp_X = mxp::adapt<2, double>(X.data(), mn);
  /* */ auto mxp_Y = mxp::adapt<2, double>(Y.data(), mn);

  gt::launch<2>(
    {mn[0], mn[1]}, GT_LAMBDA(int j, int k) {
      mxp_Y(j, k) =
        mxp_Y(j, k) + a * mxp_X(j, k) + a * mxp_X(j, k) + a * mxp_X(j, k);
    });

  EXPECT_EQ(Y[0], y_init + x_init);
  EXPECT_EQ(Y[1], y_init + x_init);
  EXPECT_EQ(Y[2], y_init + x_init);
  EXPECT_EQ(Y[3], y_init + x_init);
  EXPECT_EQ(Y[4], y_init + x_init);
  EXPECT_EQ(Y[5], y_init + x_init);
}

TEST(mxp, complex_axaxaxpy_implicit)
{
  using complex32_t = gt::complex<float>;
  using complex64_t = gt::complex<double>;

  const int n{2};
  const complex32_t x_init{exp2f(-23), -exp2f(-24)};
  const complex32_t y_init{1.f, 1.f};
  const float a_init{1.f / 3.f};

  EXPECT_NE(y_init.real(), y_init.real() + x_init.real());
  EXPECT_NE(y_init.imag(), y_init.imag() + x_init.imag());

  const std::vector<float> a(n, a_init);
  const std::vector<complex32_t> x(n, x_init);
  /* */ std::vector<complex32_t> y(n, y_init);

  const auto gt_a = gt::adapt<1>(a.data(), a.size());
  const auto gt_x = gt::adapt<1>(x.data(), x.size());
  /* */ auto gt_y = gt::adapt<1>(y.data(), y.size());

  gt_y = gt_y + gt_a * gt_x + gt_a * gt_x + gt_a * gt_x;

  EXPECT_EQ(y[0], y_init);
  EXPECT_EQ(y[1], y_init);

  const auto mxp_a = mxp::adapt<1, double>(a.data(), a.size());
  const auto mxp_x = mxp::adapt<1, complex64_t>(x.data(), x.size());
  /* */ auto mxp_y = mxp::adapt<1, complex64_t>(y.data(), y.size());

  mxp_y = mxp_y + mxp_a * mxp_x + mxp_a * mxp_x + mxp_a * mxp_x;

  EXPECT_EQ(y[0], y_init + x_init);
  EXPECT_EQ(y[1], y_init + x_init);
}

TEST(mxp, complex_axaxaxpy_explicit)
{
  using complex32_t = gt::complex<float>;
  using complex64_t = gt::complex<double>;

  const int n{2};
  const complex32_t x_init{exp2f(-23), -exp2f(-24)};
  const complex32_t y_init{1.f, 1.f};
  const float a_init{1.f / 3.f};

  EXPECT_NE(y_init.real(), y_init.real() + x_init.real());
  EXPECT_NE(y_init.imag(), y_init.imag() + x_init.imag());

  const std::vector<float> a(n, a_init);
  const std::vector<complex32_t> x(n, x_init);
  /* */ std::vector<complex32_t> y(n, y_init);

  const auto gt_a = gt::adapt<1>(a.data(), a.size());
  const auto gt_x = gt::adapt<1>(x.data(), x.size());
  /* */ auto gt_y = gt::adapt<1>(y.data(), y.size());

  gt::launch<1>(
    {n}, GT_LAMBDA(int j) {
      gt_y(j) =
        gt_y(j) + gt_a(j) * gt_x(j) + gt_a(j) * gt_x(j) + gt_a(j) * gt_x(j);
    });

  EXPECT_EQ(y[0].real(), y_init.real());
  EXPECT_EQ(y[1].real(), y_init.real());

  const auto mxp_a = mxp::adapt<1, double>(a.data(), a.size());
  const auto mxp_x = mxp::adapt<1, complex64_t>(x.data(), x.size());
  /* */ auto mxp_y = mxp::adapt<1, complex64_t>(y.data(), y.size());

  gt::launch<1>(
    {n}, GT_LAMBDA(int j) {
      mxp_y(j) = mxp_y(j) + mxp_a(j) * mxp_x(j) + mxp_a(j) * mxp_x(j) +
                 mxp_a(j) * mxp_x(j);
    });

  EXPECT_EQ(y[0], y_init + x_init);
  EXPECT_EQ(y[1], y_init + x_init);
}

TEST(mxp, complex_op_plus_implicit)
{
  using complex32_t = gt::complex<float>;
  using complex64_t = gt::complex<double>;

  const int n{2};
  const complex32_t x_init{exp2f(-23) / 3.f, -exp2f(-24) / 3.f};
  const complex32_t y_init{1.f, 1.f};

  const complex32_t gt_ref{y_init};
  const complex32_t mxp_ref{y_init + 3.f * x_init};

  EXPECT_NE(gt_ref.real(), mxp_ref.real());
  EXPECT_NE(gt_ref.imag(), mxp_ref.imag());

  const std::vector<complex32_t> x(n, x_init);
  /* */ std::vector<complex32_t> y(n, y_init);

  const auto gt_x = gt::adapt<1>(x.data(), x.size());
  /* */ auto gt_y = gt::adapt<1>(y.data(), y.size());

  gt_y = gt_y + gt_x + gt_x + gt_x;

  EXPECT_EQ(y[0], gt_ref);
  EXPECT_EQ(y[1], gt_ref);

  const auto mxp_x = mxp::adapt<1, complex64_t>(x.data(), x.size());
  /* */ auto mxp_y = mxp::adapt<1, complex64_t>(y.data(), y.size());

  mxp_y = mxp_y + mxp_x + mxp_x + mxp_x;

  EXPECT_EQ(y[0], mxp_ref);
  EXPECT_EQ(y[1], mxp_ref);
}

TEST(mxp, complex_op_minus_implicit)
{
  using complex32_t = gt::complex<float>;
  using complex64_t = gt::complex<double>;

  const int n{2};
  const complex32_t x_init{-exp2f(-23) / 3.f, exp2f(-24) / 3.f};
  const complex32_t y_init{1.f, 1.f};

  const complex32_t gt_ref{y_init};
  const complex32_t mxp_ref{y_init - 3.f * x_init};

  EXPECT_NE(gt_ref.real(), mxp_ref.real());
  EXPECT_NE(gt_ref.imag(), mxp_ref.imag());

  const std::vector<complex32_t> x(n, x_init);
  /* */ std::vector<complex32_t> y(n, y_init);

  const auto gt_x = gt::adapt<1>(x.data(), x.size());
  /* */ auto gt_y = gt::adapt<1>(y.data(), y.size());

  gt_y = gt_y - gt_x - gt_x - gt_x;

  EXPECT_EQ(y[0], gt_ref);
  EXPECT_EQ(y[1], gt_ref);

  const auto mxp_x = mxp::adapt<1, complex64_t>(x.data(), x.size());
  /* */ auto mxp_y = mxp::adapt<1, complex64_t>(y.data(), y.size());

  mxp_y = mxp_y - mxp_x - mxp_x - mxp_x;

  EXPECT_EQ(y[0], mxp_ref);
  EXPECT_EQ(y[1], mxp_ref);
}

TEST(mxp, complex_op_multiply_implicit)
{
  using complex32_t = gt::complex<float>;
  using complex64_t = gt::complex<double>;

  const int n{2};
  const complex32_t x_init{1.f + exp2f(-12), 0.f};

  const complex32_t gt_ref{1.f + exp2f(-11) + exp2f(-12) + exp2f(-23), 0.f};
  const complex32_t mxp_ref{1.f + exp2f(-11) + exp2f(-12) + exp2f(-22), 0.f};

  EXPECT_NE(gt_ref.real(), mxp_ref.real());

  const std::vector<complex32_t> x(n, x_init);
  /* */ std::vector<complex32_t> y(n);

  const auto gt_x = gt::adapt<1>(x.data(), x.size());
  /* */ auto gt_y = gt::adapt<1>(y.data(), y.size());

  gt_y = gt_x * gt_x * gt_x;

  EXPECT_EQ(y[0], gt_ref);
  EXPECT_EQ(y[1], gt_ref);

  const auto mxp_x = mxp::adapt<1, complex64_t>(x.data(), x.size());
  /* */ auto mxp_y = mxp::adapt<1, complex64_t>(y.data(), y.size());

  mxp_y = mxp_x * mxp_x * mxp_x;

  EXPECT_EQ(y[0], mxp_ref);
  EXPECT_EQ(y[1], mxp_ref);
}

TEST(mxp, complex_op_divide_implicit)
{
  using complex32_t = gt::complex<float>;
  using complex64_t = gt::complex<double>;

  const int n{2};
  double val = 1.5 + exp2f(-8) + exp2f(-15) + exp2f(-23);
  double invval = 1. / val;
  double ref = val / invval / invval;

  const complex32_t x_init{float(invval), 0.f};
  const complex32_t y_init{float(val), 0.f};

  const std::vector<complex32_t> x(n, x_init);
  /* */ std::vector<complex32_t> y_a(n, y_init);
  /* */ std::vector<complex32_t> y_b(n, y_init);

  const auto gt_x = gt::adapt<1>(x.data(), x.size());
  /* */ auto gt_y = gt::adapt<1>(y_a.data(), y_a.size());

  gt_y = gt_y / gt_x / gt_x;

  double gt_err = std::abs(y_a[1].real() - ref);
  EXPECT_GT(gt_err, 2.7e-07);

  const auto mxp_x = mxp::adapt<1, complex64_t>(x.data(), x.size());
  /* */ auto mxp_y = mxp::adapt<1, complex64_t>(y_b.data(), y_b.size());

  mxp_y = mxp_y / mxp_x / mxp_x;

  double mxp_err = std::abs(y_b[1].real() - ref);

  EXPECT_LT(mxp_err, 4.0e-08);
}
