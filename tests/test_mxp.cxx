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

  const complex32_t mxp_ref{y_init + 3.f * x_init};

  EXPECT_NE(y_init.real(), mxp_ref.real());
  EXPECT_NE(y_init.imag(), mxp_ref.imag());

  const std::vector<complex32_t> x(n, x_init);
  /* */ std::vector<complex32_t> y(n, y_init);

  const auto gt_x = gt::adapt<1>(x.data(), x.size());
  /* */ auto gt_y = gt::adapt<1>(y.data(), y.size());

  gt_y = gt_y + gt_x + gt_x + gt_x;

  EXPECT_EQ(y[0], y_init);
  EXPECT_EQ(y[1], y_init);

  const auto mxp_x = mxp::adapt<1, complex64_t>(x.data(), x.size());
  /* */ auto mxp_y = mxp::adapt<1, complex64_t>(y.data(), y.size());

  mxp_y = mxp_y + mxp_x + mxp_x + mxp_x;

  EXPECT_EQ(y[0], mxp_ref);
  EXPECT_EQ(y[1], mxp_ref);
}

TEST(mxp, complex_op_plus_explicit)
{
  using complex32_t = gt::complex<float>;
  using complex64_t = gt::complex<double>;

  const int n{2};
  const complex32_t x_init{exp2f(-23) / 3.f, -exp2f(-24) / 3.f};
  const complex32_t y_init{1.f, 1.f};

  const complex32_t mxp_ref{y_init + 3.f * x_init};

  EXPECT_NE(y_init.real(), mxp_ref.real());
  EXPECT_NE(y_init.imag(), mxp_ref.imag());

  const std::vector<complex32_t> x(n, x_init);
  /* */ std::vector<complex32_t> y(n, y_init);

  const auto gt_x = gt::adapt<1>(x.data(), x.size());
  /* */ auto gt_y = gt::adapt<1>(y.data(), y.size());

  gt::launch<1>(
    {n}, GT_LAMBDA(int j) { gt_y(j) = gt_y(j) + gt_x(j) + gt_x(j) + gt_x(j); });

  EXPECT_EQ(y[0], y_init);
  EXPECT_EQ(y[1], y_init);

  const auto mxp_x = mxp::adapt<1, complex64_t>(x.data(), x.size());
  /* */ auto mxp_y = mxp::adapt<1, complex64_t>(y.data(), y.size());

  gt::launch<1>(
    {n},
    GT_LAMBDA(int j) { mxp_y(j) = mxp_y(j) + mxp_x(j) + mxp_x(j) + mxp_x(j); });

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

  const complex32_t mxp_ref{y_init - 3.f * x_init};

  EXPECT_NE(y_init.real(), mxp_ref.real());
  EXPECT_NE(y_init.imag(), mxp_ref.imag());

  const std::vector<complex32_t> x(n, x_init);
  /* */ std::vector<complex32_t> y(n, y_init);

  const auto gt_x = gt::adapt<1>(x.data(), x.size());
  /* */ auto gt_y = gt::adapt<1>(y.data(), y.size());

  gt_y = gt_y - gt_x - gt_x - gt_x;

  EXPECT_EQ(y[0], y_init);
  EXPECT_EQ(y[1], y_init);

  const auto mxp_x = mxp::adapt<1, complex64_t>(x.data(), x.size());
  /* */ auto mxp_y = mxp::adapt<1, complex64_t>(y.data(), y.size());

  mxp_y = mxp_y - mxp_x - mxp_x - mxp_x;

  EXPECT_EQ(y[0], mxp_ref);
  EXPECT_EQ(y[1], mxp_ref);
}

TEST(mxp, complex_op_minus_explicit)
{
  using complex32_t = gt::complex<float>;
  using complex64_t = gt::complex<double>;

  const int n{2};
  const complex32_t x_init{-exp2f(-23) / 3.f, exp2f(-24) / 3.f};
  const complex32_t y_init{1.f, 1.f};

  const complex32_t mxp_ref{y_init - 3.f * x_init};

  EXPECT_NE(y_init.real(), mxp_ref.real());
  EXPECT_NE(y_init.imag(), mxp_ref.imag());

  const std::vector<complex32_t> x(n, x_init);
  /* */ std::vector<complex32_t> y(n, y_init);

  const auto gt_x = gt::adapt<1>(x.data(), x.size());
  /* */ auto gt_y = gt::adapt<1>(y.data(), y.size());

  gt::launch<1>(
    {n}, GT_LAMBDA(int j) { gt_y(j) = gt_y(j) - gt_x(j) - gt_x(j) - gt_x(j); });

  EXPECT_EQ(y[0], y_init);
  EXPECT_EQ(y[1], y_init);

  const auto mxp_x = mxp::adapt<1, complex64_t>(x.data(), x.size());
  /* */ auto mxp_y = mxp::adapt<1, complex64_t>(y.data(), y.size());

  gt::launch<1>(
    {n},
    GT_LAMBDA(int j) { mxp_y(j) = mxp_y(j) - mxp_x(j) - mxp_x(j) - mxp_x(j); });

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

TEST(mxp, complex_op_multiply_explicit)
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

  gt::launch<1>(
    {n}, GT_LAMBDA(int j) { gt_y(j) = gt_x(j) * gt_x(j) * gt_x(j); });

  EXPECT_EQ(y[0], gt_ref);
  EXPECT_EQ(y[1], gt_ref);

  const auto mxp_x = mxp::adapt<1, complex64_t>(x.data(), x.size());
  /* */ auto mxp_y = mxp::adapt<1, complex64_t>(y.data(), y.size());

  gt::launch<1>(
    {n}, GT_LAMBDA(int j) { mxp_y(j) = mxp_x(j) * mxp_x(j) * mxp_x(j); });

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

TEST(mxp, complex_op_divide_explicit)
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

  gt::launch<1>(
    {n}, GT_LAMBDA(int j) { gt_y(j) = gt_y(j) / gt_x(j) / gt_x(j); });

  double gt_err = std::abs(y_a[1].real() - ref);
  EXPECT_GT(gt_err, 2.7e-07);

  const auto mxp_x = mxp::adapt<1, complex64_t>(x.data(), x.size());
  /* */ auto mxp_y = mxp::adapt<1, complex64_t>(y_b.data(), y_b.size());

  gt::launch<1>(
    {n}, GT_LAMBDA(int j) { mxp_y(j) = mxp_y(j) / mxp_x(j) / mxp_x(j); });

  double mxp_err = std::abs(y_b[1].real() - ref);

  EXPECT_LT(mxp_err, 4.0e-08);
}

TEST(mxp, view_axaxaxpy)
{
  const int nx{3};
  const int ny{5};
  const float x_init{exp2f(-23)};
  const float y_init{1.f};
  const float a{1.f / 3.f};

  EXPECT_NE(y_init, y_init + x_init);

  const std::vector<float> x(nx, x_init);
  /* */ std::vector<float> y(ny, y_init);

  const auto gt_x = gt::adapt<1>(x.data(), x.size());
  /* */ auto gt_y = gt::adapt<1>(y.data(), y.size());

  using gt::placeholders::_all;
  using gt::placeholders::_s;

  gt_y.view(_s(1, -1)) = gt_y.view(_s(1, -1)) + a * gt_x.view(_all) +
                         a * gt_x.view(_all) + a * gt_x.view(_all);

  EXPECT_EQ(y[0], y_init);
  EXPECT_EQ(y[1], y_init);
  EXPECT_EQ(y[2], y_init);
  EXPECT_EQ(y[3], y_init);
  EXPECT_EQ(y[4], y_init);

  const auto mxp_x = mxp::adapt<1, double>(x.data(), x.size());
  /* */ auto mxp_y = mxp::adapt<1, double>(y.data(), y.size());

  mxp_y.view(_s(1, -1)) = mxp_y.view(_s(1, -1)) + a * mxp_x.view(_all) +
                          a * mxp_x.view(_all) + a * mxp_x.view(_all);

  EXPECT_EQ(y[0], y_init);
  EXPECT_EQ(y[1], y_init + x_init);
  EXPECT_EQ(y[2], y_init + x_init);
  EXPECT_EQ(y[3], y_init + x_init);
  EXPECT_EQ(y[4], y_init);
}

TEST(mxp, view_all_2D)
{
  const int mn[2]{3, 2};
  const float x_init{exp2f(-23)};
  const float y_init{1.f};
  const float a{1.f / 3.f};

  EXPECT_NE(y_init, y_init + x_init);

  const std::vector<float> X(mn[0] * mn[1], x_init);
  /* */ std::vector<float> Y(mn[0] * mn[1], y_init);

  const auto gt_X = gt::adapt<2>(X.data(), mn);
  /* */ auto gt_Y = gt::adapt<2>(Y.data(), mn);

  using gt::placeholders::_all;

  gt_Y.view(_all, _all) = gt_Y.view(_all, _all) + a * gt_X.view(_all, _all) +
                          a * gt_X.view(_all, _all) + a * gt_X.view(_all, _all);

  EXPECT_EQ(Y[0], y_init);
  EXPECT_EQ(Y[1], y_init);
  EXPECT_EQ(Y[2], y_init);
  EXPECT_EQ(Y[3], y_init);
  EXPECT_EQ(Y[4], y_init);
  EXPECT_EQ(Y[5], y_init);

  const auto mxp_X = mxp::adapt<2, double>(X.data(), mn);
  /* */ auto mxp_Y = mxp::adapt<2, double>(Y.data(), mn);

  mxp_Y.view(_all, _all) = mxp_Y.view(_all, _all) + a * mxp_X.view(_all, _all) +
                           a * mxp_X.view(_all, _all) +
                           a * mxp_X.view(_all, _all);

  EXPECT_EQ(Y[0], y_init + x_init);
  EXPECT_EQ(Y[1], y_init + x_init);
  EXPECT_EQ(Y[2], y_init + x_init);
  EXPECT_EQ(Y[3], y_init + x_init);
  EXPECT_EQ(Y[4], y_init + x_init);
  EXPECT_EQ(Y[5], y_init + x_init);
}

TEST(mxp, view_newaxis_2D)
{
  const int mn[2]{3, 2};
  const float x_init{exp2f(-23)};
  const float y_init{1.f};
  const float a{1.f / 3.f};

  EXPECT_NE(y_init, y_init + x_init);

  const std::vector<float> x(mn[0], x_init);
  /* */ std::vector<float> Y(mn[0] * mn[1], y_init);

  const auto gt_x = gt::adapt<1>(x.data(), mn[0]);
  /* */ auto gt_Y = gt::adapt<2>(Y.data(), mn);

  using gt::placeholders::_all;
  using gt::placeholders::_newaxis;

  gt_Y.view(_all, _all) =
    gt_Y.view(_all, _all) + a * gt_x.view(_all, _newaxis) +
    a * gt_x.view(_all, _newaxis) + a * gt_x.view(_all, _newaxis);

  EXPECT_EQ(Y[0], y_init);
  EXPECT_EQ(Y[1], y_init);
  EXPECT_EQ(Y[2], y_init);
  EXPECT_EQ(Y[3], y_init);
  EXPECT_EQ(Y[4], y_init);
  EXPECT_EQ(Y[5], y_init);

  const auto mxp_x = mxp::adapt<1, double>(x.data(), mn[0]);
  /* */ auto mxp_Y = mxp::adapt<2, double>(Y.data(), mn);

  mxp_Y.view(_all, _all) =
    mxp_Y.view(_all, _all) + a * mxp_x.view(_all, _newaxis) +
    a * mxp_x.view(_all, _newaxis) + a * mxp_x.view(_all, _newaxis);

  EXPECT_EQ(Y[0], y_init + x_init);
  EXPECT_EQ(Y[1], y_init + x_init);
  EXPECT_EQ(Y[2], y_init + x_init);
  EXPECT_EQ(Y[3], y_init + x_init);
  EXPECT_EQ(Y[4], y_init + x_init);
  EXPECT_EQ(Y[5], y_init + x_init);
}

TEST(mxp, view_s_2D)
{
  const int mn[2]{4, 3};
  const int lj = 2, uj = 4, lk = 1, uk = 3;

  const float x_init{exp2f(-23)};
  const float y_init{1.f};
  const float a{1.f / 3.f};

  EXPECT_NE(y_init, y_init + x_init);

  const std::vector<float> X(mn[0] * mn[1], x_init);
  /* */ std::vector<float> Y(mn[0] * mn[1], y_init);

  const auto gt_X = gt::adapt<2>(X.data(), mn);
  /* */ auto gt_Y = gt::adapt<2>(Y.data(), mn);

  using gt::placeholders::_s;

  gt_Y.view(_s(lj, uj), _s(lk, uk)) = gt_Y.view(_s(lj, uj), _s(lk, uk)) +
                                      a * gt_X.view(_s(lj, uj), _s(lk, uk)) +
                                      a * gt_X.view(_s(lj, uj), _s(lk, uk)) +
                                      a * gt_X.view(_s(lj, uj), _s(lk, uk));

  for (int j = 0; j < mn[0]; ++j) {
    for (int k = 0; k < mn[1]; ++k) {
      const int idx = j + k * mn[0];
      EXPECT_EQ(Y[idx], y_init);
    }
  }

  const auto mxp_X = mxp::adapt<2, double>(X.data(), mn);
  /* */ auto mxp_Y = mxp::adapt<2, double>(Y.data(), mn);

  mxp_Y.view(_s(lj, uj), _s(lk, uk)) = mxp_Y.view(_s(lj, uj), _s(lk, uk)) +
                                       a * mxp_X.view(_s(lj, uj), _s(lk, uk)) +
                                       a * mxp_X.view(_s(lj, uj), _s(lk, uk)) +
                                       a * mxp_X.view(_s(lj, uj), _s(lk, uk));

  for (int j = 0; j < mn[0]; ++j) {
    for (int k = 0; k < mn[1]; ++k) {
      const int idx = j + k * mn[0];

      if (j >= lj && j < uj && k >= lk && k < uk)
        EXPECT_EQ(Y[idx], y_init + x_init);
      else
        EXPECT_EQ(Y[idx], y_init);
    }
  }
}

TEST(mxp, view_slice_2D)
{
  const int mn[2]{4, 3};
  const int slice_idx = 1;

  const float x_init{exp2f(-23)};
  const float y_init{1.f};
  const float a{1.f / 3.f};

  EXPECT_NE(y_init, y_init + x_init);

  const std::vector<float> X(mn[0] * mn[1], x_init);
  /* */ std::vector<float> Y(mn[0] * mn[1], y_init);

  const auto gt_X = gt::adapt<2>(X.data(), mn);
  /* */ auto gt_Y = gt::adapt<2>(Y.data(), mn);

  using gt::placeholders::_all;

  gt_Y.view(_all, slice_idx) =
    gt_Y.view(_all, slice_idx) + a * gt_X.view(_all, slice_idx) +
    a * gt_X.view(_all, slice_idx) + a * gt_X.view(_all, slice_idx);

  for (int j = 0; j < mn[0]; ++j) {
    for (int k = 0; k < mn[1]; ++k) {
      const int idx = j + k * mn[0];
      EXPECT_EQ(Y[idx], y_init);
    }
  }

  const auto mxp_X = mxp::adapt<2, double>(X.data(), mn);
  /* */ auto mxp_Y = mxp::adapt<2, double>(Y.data(), mn);

  mxp_Y.view(_all, slice_idx) =
    mxp_Y.view(_all, slice_idx) + a * mxp_X.view(_all, slice_idx) +
    a * mxp_X.view(_all, slice_idx) + a * mxp_X.view(_all, slice_idx);

  for (int j = 0; j < mn[0]; ++j) {
    for (int k = 0; k < mn[1]; ++k) {
      const int idx = j + k * mn[0];

      if (k == slice_idx)
        EXPECT_EQ(Y[idx], y_init + x_init);
      else
        EXPECT_EQ(Y[idx], y_init);
    }
  }
}

TEST(mxp, view_complex_axaxaxpy)
{
  using complex32_t = gt::complex<float>;
  using complex64_t = gt::complex<double>;

  const int nx{3};
  const int ny{5};
  const complex32_t x_init{exp2f(-23), -exp2f(-24)};
  const complex32_t y_init{1.f, 1.f};
  const float a_init{1.f / 3.f};

  EXPECT_NE(y_init.real(), y_init.real() + x_init.real());
  EXPECT_NE(y_init.imag(), y_init.imag() + x_init.imag());

  const std::vector<float> a(nx, a_init);
  const std::vector<complex32_t> x(nx, x_init);
  /* */ std::vector<complex32_t> y(ny, y_init);

  const auto gt_a = gt::adapt<1>(a.data(), a.size());
  const auto gt_x = gt::adapt<1>(x.data(), x.size());
  /* */ auto gt_y = gt::adapt<1>(y.data(), y.size());

  using gt::placeholders::_all;
  using gt::placeholders::_s;

  gt_y.view(_s(1, -1)) =
    gt_y.view(_s(1, -1)) + gt_a.view(_all) * gt_x.view(_all) +
    gt_a.view(_all) * gt_x.view(_all) + gt_a.view(_all) * gt_x.view(_all);

  EXPECT_EQ(y[0], y_init);
  EXPECT_EQ(y[1], y_init);
  EXPECT_EQ(y[2], y_init);
  EXPECT_EQ(y[3], y_init);
  EXPECT_EQ(y[4], y_init);

  const auto mxp_a = mxp::adapt<1, double>(a.data(), a.size());
  const auto mxp_x = mxp::adapt<1, complex64_t>(x.data(), x.size());
  /* */ auto mxp_y = mxp::adapt<1, complex64_t>(y.data(), y.size());

  mxp_y.view(_s(1, -1)) =
    mxp_y.view(_s(1, -1)) + mxp_a.view(_all) * mxp_x.view(_all) +
    mxp_a.view(_all) * mxp_x.view(_all) + mxp_a.view(_all) * mxp_x.view(_all);

  EXPECT_EQ(y[0], y_init);
  EXPECT_EQ(y[1], y_init + x_init);
  EXPECT_EQ(y[2], y_init + x_init);
  EXPECT_EQ(y[3], y_init + x_init);
  EXPECT_EQ(y[4], y_init);
}

TEST(mxp, view_complex_op_plus)
{
  using complex32_t = gt::complex<float>;
  using complex64_t = gt::complex<double>;

  const int nx{3};
  const int ny{5};
  const complex32_t x_init{exp2f(-23) / 3.f, -exp2f(-24) / 3.f};
  const complex32_t y_init{1.f, 1.f};

  const complex32_t mxp_ref{y_init + 3.f * x_init};

  EXPECT_NE(y_init.real(), mxp_ref.real());
  EXPECT_NE(y_init.imag(), mxp_ref.imag());

  const std::vector<complex32_t> x(nx, x_init);
  /* */ std::vector<complex32_t> y(ny, y_init);

  const auto gt_x = gt::adapt<1>(x.data(), x.size());
  /* */ auto gt_y = gt::adapt<1>(y.data(), y.size());

  using gt::placeholders::_all;
  using gt::placeholders::_s;

  gt_y.view(_s(1, -1)) =
    gt_y.view(_s(1, -1)) + gt_x.view(_all) + gt_x.view(_all) + gt_x.view(_all);

  EXPECT_EQ(y[0], y_init);
  EXPECT_EQ(y[1], y_init);
  EXPECT_EQ(y[2], y_init);
  EXPECT_EQ(y[3], y_init);
  EXPECT_EQ(y[4], y_init);

  const auto mxp_x = mxp::adapt<1, complex64_t>(x.data(), x.size());
  /* */ auto mxp_y = mxp::adapt<1, complex64_t>(y.data(), y.size());

  mxp_y.view(_s(1, -1)) = mxp_y.view(_s(1, -1)) + mxp_x.view(_all) +
                          mxp_x.view(_all) + mxp_x.view(_all);

  EXPECT_EQ(y[0], y_init);
  EXPECT_EQ(y[1], mxp_ref);
  EXPECT_EQ(y[2], mxp_ref);
  EXPECT_EQ(y[3], mxp_ref);
  EXPECT_EQ(y[4], y_init);
}

TEST(mxp, view_complex_op_minus)
{
  using complex32_t = gt::complex<float>;
  using complex64_t = gt::complex<double>;

  const int nx{3};
  const int ny{5};
  const complex32_t x_init{-exp2f(-23) / 3.f, exp2f(-24) / 3.f};
  const complex32_t y_init{1.f, 1.f};

  const complex32_t mxp_ref{y_init - 3.f * x_init};

  EXPECT_NE(y_init.real(), mxp_ref.real());
  EXPECT_NE(y_init.imag(), mxp_ref.imag());

  const std::vector<complex32_t> x(nx, x_init);
  /* */ std::vector<complex32_t> y(ny, y_init);

  const auto gt_x = gt::adapt<1>(x.data(), x.size());
  /* */ auto gt_y = gt::adapt<1>(y.data(), y.size());

  using gt::placeholders::_all;
  using gt::placeholders::_s;

  gt_y.view(_s(1, -1)) =
    gt_y.view(_s(1, -1)) - gt_x.view(_all) - gt_x.view(_all) - gt_x.view(_all);

  EXPECT_EQ(y[0], y_init);
  EXPECT_EQ(y[1], y_init);
  EXPECT_EQ(y[2], y_init);
  EXPECT_EQ(y[3], y_init);
  EXPECT_EQ(y[4], y_init);

  const auto mxp_x = mxp::adapt<1, complex64_t>(x.data(), x.size());
  /* */ auto mxp_y = mxp::adapt<1, complex64_t>(y.data(), y.size());

  mxp_y.view(_s(1, -1)) = mxp_y.view(_s(1, -1)) - mxp_x.view(_all) -
                          mxp_x.view(_all) - mxp_x.view(_all);

  EXPECT_EQ(y[0], y_init);
  EXPECT_EQ(y[1], mxp_ref);
  EXPECT_EQ(y[2], mxp_ref);
  EXPECT_EQ(y[3], mxp_ref);
  EXPECT_EQ(y[4], y_init);
}

TEST(mxp, view_complex_op_multiply)
{
  using complex32_t = gt::complex<float>;
  using complex64_t = gt::complex<double>;

  const int nx{3};
  const int ny{5};
  const complex32_t x_init{1.f + exp2f(-12), 0.f};
  const complex32_t y_init{-654.321f};

  const complex32_t gt_ref{1.f + exp2f(-11) + exp2f(-12) + exp2f(-23), 0.f};
  const complex32_t mxp_ref{1.f + exp2f(-11) + exp2f(-12) + exp2f(-22), 0.f};

  EXPECT_NE(gt_ref.real(), mxp_ref.real());

  const std::vector<complex32_t> x(nx, x_init);
  /* */ std::vector<complex32_t> y(ny, y_init);

  const auto gt_x = gt::adapt<1>(x.data(), x.size());
  /* */ auto gt_y = gt::adapt<1>(y.data(), y.size());

  using gt::placeholders::_all;
  using gt::placeholders::_s;

  gt_y.view(_s(1, -1)) = gt_x.view(_all) * gt_x.view(_all) * gt_x.view(_all);

  EXPECT_EQ(y[0], y_init);
  EXPECT_EQ(y[1], gt_ref);
  EXPECT_EQ(y[2], gt_ref);
  EXPECT_EQ(y[3], gt_ref);
  EXPECT_EQ(y[4], y_init);

  const auto mxp_x = mxp::adapt<1, complex64_t>(x.data(), x.size());
  /* */ auto mxp_y = mxp::adapt<1, complex64_t>(y.data(), y.size());

  mxp_y.view(_s(1, -1)) =
    mxp_x.view(_all) * mxp_x.view(_all) * mxp_x.view(_all);

  EXPECT_EQ(y[0], y_init);
  EXPECT_EQ(y[1], mxp_ref);
  EXPECT_EQ(y[2], mxp_ref);
  EXPECT_EQ(y[3], mxp_ref);
  EXPECT_EQ(y[4], y_init);
}

TEST(mxp, view_complex_op_divide)
{
  using complex32_t = gt::complex<float>;
  using complex64_t = gt::complex<double>;

  const int nx{3};
  const int ny{5};
  double val = 1.5 + exp2f(-8) + exp2f(-15) + exp2f(-23);
  double invval = 1. / val;
  double ref = val / invval / invval;

  const complex32_t x_init{float(invval), 0.f};
  const complex32_t y_init{float(val), 0.f};

  const std::vector<complex32_t> x(nx, x_init);
  /* */ std::vector<complex32_t> y_a(ny, y_init);
  /* */ std::vector<complex32_t> y_b(ny, y_init);

  const auto gt_x = gt::adapt<1>(x.data(), x.size());
  /* */ auto gt_y = gt::adapt<1>(y_a.data(), y_a.size());

  using gt::placeholders::_all;
  using gt::placeholders::_s;

  gt_y.view(_s(1, -1)) =
    gt_y.view(_s(1, -1)) / gt_x.view(_all) / gt_x.view(_all);

  double gt_err = std::abs(y_a[1].real() - ref);

  EXPECT_EQ(y_a[0], y_init);
  EXPECT_GT(gt_err, 2.7e-07);
  EXPECT_EQ(y_a[2], y_a[1]);
  EXPECT_EQ(y_a[3], y_a[1]);
  EXPECT_EQ(y_a[4], y_init);

  const auto mxp_x = mxp::adapt<1, complex64_t>(x.data(), x.size());
  /* */ auto mxp_y = mxp::adapt<1, complex64_t>(y_b.data(), y_b.size());

  mxp_y.view(_s(1, -1)) =
    mxp_y.view(_s(1, -1)) / mxp_x.view(_all) / mxp_x.view(_all);

  double mxp_err = std::abs(y_b[1].real() - ref);

  EXPECT_EQ(y_b[0], y_init);
  EXPECT_LT(mxp_err, 4.0e-08);
  EXPECT_EQ(y_b[2], y_b[1]);
  EXPECT_EQ(y_b[3], y_b[1]);
  EXPECT_EQ(y_b[4], y_init);
}
