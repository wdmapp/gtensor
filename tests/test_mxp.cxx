#include <gtest/gtest.h>

#include <gtensor/gtensor.h>

#include <gtensor/mxp.h>

#include <cmath>

TEST(mxp, axaxaxpy_implicit)
{
  const int n{2};
  const float x_init{exp2f(-23)};
  const float y_init{1.f};
  const float a{1.f / 3.f};

  EXPECT_NE(y_init, y_init + x_init);

  const gt::gtensor<float, 1> x(n, x_init);
  /* */ gt::gtensor<float, 1> y(n, y_init);

  const auto gt_x = gt::adapt<1>(x.data(), x.size());
  /* */ auto gt_y = gt::adapt<1>(y.data(), y.size());

  gt_y = gt_y + a * gt_x + a * gt_x + a * gt_x;

  EXPECT_EQ(y, (gt::gtensor<float, 1>(n, y_init)));

  const auto mxp_x = mxp::adapt<1, double>(x.data(), x.size());
  /* */ auto mxp_y = mxp::adapt<1, double>(y.data(), y.size());

  mxp_y = mxp_y + a * mxp_x + a * mxp_x + a * mxp_x;

  EXPECT_EQ(y, (gt::gtensor<float, 1>(n, y_init + x_init)));
}

template <typename S, typename T>
void generic_axaxaxpy_explicit_gt(const int n, const T a,
                                  const gt::gtensor<T, 1, S>& x,
                                  gt::gtensor<T, 1, S>& y)
{
  const auto gt_x = gt::adapt<1, S>(x.data(), n);
  /* */ auto gt_y = gt::adapt<1, S>(y.data(), n);

  gt::launch<1, S>(
    {n}, GT_LAMBDA(int j) {
      gt_y(j) = gt_y(j) + a * gt_x(j) + a * gt_x(j) + a * gt_x(j);
    });
}

template <typename S, typename X, typename T>
void generic_axaxaxpy_explicit_mxp(const int n, const T a,
                                   const gt::gtensor<T, 1, S>& x,
                                   gt::gtensor<T, 1, S>& y)
{
  const auto mxp_x = mxp::adapt<1, S, X>(x.data(), n);
  /* */ auto mxp_y = mxp::adapt<1, S, X>(y.data(), n);

  gt::launch<1, S>(
    {n}, GT_LAMBDA(int j) {
      mxp_y(j) = mxp_y(j) + a * mxp_x(j) + a * mxp_x(j) + a * mxp_x(j);
    });
}

TEST(mxp, axaxaxpy_explicit)
{
  const int n{2};
  const float x_init{exp2f(-23)};
  const float y_init{1.f};
  const float a{1.f / 3.f};

  EXPECT_NE(y_init, y_init + x_init);

  const gt::gtensor<float, 1> x(n, x_init);
  /* */ gt::gtensor<float, 1> y(n, y_init);

  generic_axaxaxpy_explicit_gt<gt::space::host>(n, a, x, y);

  EXPECT_EQ(y, (gt::gtensor<float, 1>(n, y_init)));

  generic_axaxaxpy_explicit_mxp<gt::space::host, double>(n, a, x, y);

  EXPECT_EQ(y, (gt::gtensor<float, 1>(n, y_init + x_init)));
}

TEST(mxp, aXaXaXpY_2D_implicit)
{
  const int mn[2]{2, 3};
  const float x_init{exp2f(-23)};
  const float y_init{1.f};
  const float a{1.f / 3.f};

  EXPECT_NE(y_init, y_init + x_init);

  const gt::gtensor<float, 2> X(gt::shape(mn[0], mn[1]), x_init);
  /* */ gt::gtensor<float, 2> Y(gt::shape(mn[0], mn[1]), y_init);

  const auto gt_X = gt::adapt<2>(X.data(), mn);
  /* */ auto gt_Y = gt::adapt<2>(Y.data(), mn);

  gt_Y = gt_Y + a * gt_X + a * gt_X + a * gt_X;

  EXPECT_EQ(Y, (gt::gtensor<float, 2>(gt::shape(mn[0], mn[1]), y_init)));

  const auto mxp_X = mxp::adapt<2, double>(X.data(), mn);
  /* */ auto mxp_Y = mxp::adapt<2, double>(Y.data(), mn);

  mxp_Y = mxp_Y + a * mxp_X + a * mxp_X + a * mxp_X;

  EXPECT_EQ(Y,
            (gt::gtensor<float, 2>(gt::shape(mn[0], mn[1]), y_init + x_init)));
}

template <typename S, typename T>
void generic_aXaXaXpy_2D_explicit_gt(const int* mn, const T a,
                                     const gt::gtensor<T, 2, S>& xx,
                                     gt::gtensor<T, 2, S>& yy)
{
  const auto gt_X = gt::adapt<2, S>(xx.data(), mn);
  /* */ auto gt_Y = gt::adapt<2, S>(yy.data(), mn);

  gt::launch<2, S>(
    {mn[0], mn[1]}, GT_LAMBDA(int j, int k) {
      gt_Y(j, k) =
        gt_Y(j, k) + a * gt_X(j, k) + a * gt_X(j, k) + a * gt_X(j, k);
    });
}

template <typename S, typename X, typename T>
void generic_aXaXaXpy_2D_explicit_mxp(const int* mn, const T a,
                                      const gt::gtensor<T, 2, S>& xx,
                                      gt::gtensor<T, 2, S>& yy)
{
  const auto mxp_X = mxp::adapt<2, S, X>(xx.data(), mn);
  /* */ auto mxp_Y = mxp::adapt<2, S, X>(yy.data(), mn);

  gt::launch<2, S>(
    {mn[0], mn[1]}, GT_LAMBDA(int j, int k) {
      mxp_Y(j, k) =
        mxp_Y(j, k) + a * mxp_X(j, k) + a * mxp_X(j, k) + a * mxp_X(j, k);
    });
}

TEST(mxp, aXaXaXpY_2D_explicit)
{
  const int mn[2]{2, 3};
  const float x_init{exp2f(-23)};
  const float y_init{1.f};
  const float a{1.f / 3.f};

  EXPECT_NE(y_init, y_init + x_init);

  const gt::gtensor<float, 2> X(gt::shape(mn[0], mn[1]), x_init);
  /* */ gt::gtensor<float, 2> Y(gt::shape(mn[0], mn[1]), y_init);

  generic_aXaXaXpy_2D_explicit_gt<gt::space::host>(mn, a, X, Y);

  EXPECT_EQ(Y, (gt::gtensor<float, 2>(gt::shape(mn[0], mn[1]), y_init)));

  generic_aXaXaXpy_2D_explicit_mxp<gt::space::host, double>(mn, a, X, Y);

  EXPECT_EQ(Y,
            (gt::gtensor<float, 2>(gt::shape(mn[0], mn[1]), y_init + x_init)));
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

  const gt::gtensor<float, 1> a(n, a_init);
  const gt::gtensor<complex32_t, 1> x(n, x_init);
  /* */ gt::gtensor<complex32_t, 1> y(n, y_init);

  const auto gt_a = gt::adapt<1>(a.data(), a.size());
  const auto gt_x = gt::adapt<1>(x.data(), x.size());
  /* */ auto gt_y = gt::adapt<1>(y.data(), y.size());

  gt_y = gt_y + gt_a * gt_x + gt_a * gt_x + gt_a * gt_x;

  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, y_init)));

  const auto mxp_a = mxp::adapt<1, double>(a.data(), a.size());
  const auto mxp_x = mxp::adapt<1, complex64_t>(x.data(), x.size());
  /* */ auto mxp_y = mxp::adapt<1, complex64_t>(y.data(), y.size());

  mxp_y = mxp_y + mxp_a * mxp_x + mxp_a * mxp_x + mxp_a * mxp_x;

  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, y_init + x_init)));
}

template <typename S, typename T>
void generic_complex_axaxaxpy_explicit_gt(const int n, const T* a,
                                          const gt::complex<T>* x,
                                          gt::complex<T>* y)
{
  const auto gt_a = gt::adapt<1, S>(a, n);
  const auto gt_x = gt::adapt<1, S>(x, n);
  /* */ auto gt_y = gt::adapt<1, S>(y, n);

  gt::launch<1, S>(
    {n}, GT_LAMBDA(int j) {
      gt_y(j) =
        gt_y(j) + gt_a(j) * gt_x(j) + gt_a(j) * gt_x(j) + gt_a(j) * gt_x(j);
    });
}

template <typename S, typename X, typename T>
void generic_complex_axaxaxpy_explicit_mxp(const int n, const T* a,
                                           const gt::complex<T>* x,
                                           gt::complex<T>* y)
{
  const auto mxp_a = mxp::adapt<1, S, X>(a, n);
  const auto mxp_x = mxp::adapt<1, S, X>(x, n);
  /* */ auto mxp_y = mxp::adapt<1, S, X>(y, n);

  gt::launch<1, S>(
    {n}, GT_LAMBDA(int j) {
      mxp_y(j) = mxp_y(j) + mxp_a(j) * mxp_x(j) + mxp_a(j) * mxp_x(j) +
                 mxp_a(j) * mxp_x(j);
    });
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

  const gt::gtensor<float, 1> a(n, a_init);
  const gt::gtensor<complex32_t, 1> x(n, x_init);
  /* */ gt::gtensor<complex32_t, 1> y(n, y_init);

  generic_complex_axaxaxpy_explicit_gt<gt::space::host>(n, a.data(), x.data(),
                                                        y.data());

  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, y_init)));

  generic_complex_axaxaxpy_explicit_mxp<gt::space::host, complex64_t>(
    n, a.data(), x.data(), y.data());

  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, y_init + x_init)));
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

  const gt::gtensor<complex32_t, 1> x(n, x_init);
  /* */ gt::gtensor<complex32_t, 1> y(n, y_init);

  const auto gt_x = gt::adapt<1>(x.data(), x.size());
  /* */ auto gt_y = gt::adapt<1>(y.data(), y.size());

  gt_y = gt_y + gt_x + gt_x + gt_x;

  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, y_init)));

  const auto mxp_x = mxp::adapt<1, complex64_t>(x.data(), x.size());
  /* */ auto mxp_y = mxp::adapt<1, complex64_t>(y.data(), y.size());

  mxp_y = mxp_y + mxp_x + mxp_x + mxp_x;

  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, mxp_ref)));
}

template <typename S, typename T>
void generic_complex_op_plus_explicit_gt(const int n, const gt::complex<T>* x,
                                         gt::complex<T>* y)
{
  const auto gt_x = gt::adapt<1, S>(x, n);
  /* */ auto gt_y = gt::adapt<1, S>(y, n);

  gt::launch<1, S>(
    {n}, GT_LAMBDA(int j) { gt_y(j) = gt_y(j) + gt_x(j) + gt_x(j) + gt_x(j); });
}

template <typename S, typename X, typename T>
void generic_complex_op_plus_explicit_mxp(const int n, const gt::complex<T>* x,
                                          gt::complex<T>* y)
{
  const auto mxp_x = mxp::adapt<1, S, X>(x, n);
  /* */ auto mxp_y = mxp::adapt<1, S, X>(y, n);

  gt::launch<1, S>(
    {n},
    GT_LAMBDA(int j) { mxp_y(j) = mxp_y(j) + mxp_x(j) + mxp_x(j) + mxp_x(j); });
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

  const gt::gtensor<complex32_t, 1> x(n, x_init);
  /* */ gt::gtensor<complex32_t, 1> y(n, y_init);

  generic_complex_op_plus_explicit_gt<gt::space::host>(n, x.data(), y.data());

  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, y_init)));

  generic_complex_op_plus_explicit_mxp<gt::space::host, complex64_t>(
    n, x.data(), y.data());

  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, mxp_ref)));
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

  const gt::gtensor<complex32_t, 1> x(n, x_init);
  /* */ gt::gtensor<complex32_t, 1> y(n, y_init);

  const auto gt_x = gt::adapt<1>(x.data(), x.size());
  /* */ auto gt_y = gt::adapt<1>(y.data(), y.size());

  gt_y = gt_y - gt_x - gt_x - gt_x;

  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, y_init)));

  const auto mxp_x = mxp::adapt<1, complex64_t>(x.data(), x.size());
  /* */ auto mxp_y = mxp::adapt<1, complex64_t>(y.data(), y.size());

  mxp_y = mxp_y - mxp_x - mxp_x - mxp_x;

  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, mxp_ref)));
}

template <typename S, typename T>
void generic_complex_op_minus_explicit_gt(const int n, const gt::complex<T>* x,
                                          gt::complex<T>* y)
{
  const auto gt_x = gt::adapt<1, S>(x, n);
  /* */ auto gt_y = gt::adapt<1, S>(y, n);

  gt::launch<1, S>(
    {n}, GT_LAMBDA(int j) { gt_y(j) = gt_y(j) - gt_x(j) - gt_x(j) - gt_x(j); });
}

template <typename S, typename X, typename T>
void generic_complex_op_minus_explicit_mxp(const int n, const gt::complex<T>* x,
                                           gt::complex<T>* y)
{
  const auto mxp_x = mxp::adapt<1, S, X>(x, n);
  /* */ auto mxp_y = mxp::adapt<1, S, X>(y, n);

  gt::launch<1, S>(
    {n},
    GT_LAMBDA(int j) { mxp_y(j) = mxp_y(j) - mxp_x(j) - mxp_x(j) - mxp_x(j); });
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

  const gt::gtensor<complex32_t, 1> x(n, x_init);
  /* */ gt::gtensor<complex32_t, 1> y(n, y_init);

  generic_complex_op_minus_explicit_gt<gt::space::host>(n, x.data(), y.data());

  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, y_init)));

  generic_complex_op_minus_explicit_mxp<gt::space::host, complex64_t>(
    n, x.data(), y.data());

  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, mxp_ref)));
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

  const gt::gtensor<complex32_t, 1> x(n, x_init);
  /* */ gt::gtensor<complex32_t, 1> y(n);

  const auto gt_x = gt::adapt<1>(x.data(), x.size());
  /* */ auto gt_y = gt::adapt<1>(y.data(), y.size());

  gt_y = gt_x * gt_x * gt_x;

  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, gt_ref)));

  const auto mxp_x = mxp::adapt<1, complex64_t>(x.data(), x.size());
  /* */ auto mxp_y = mxp::adapt<1, complex64_t>(y.data(), y.size());

  mxp_y = mxp_x * mxp_x * mxp_x;

  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, mxp_ref)));
}

template <typename S, typename T>
void generic_complex_op_multiply_explicit_gt(const int n,
                                             const gt::complex<T>* x,
                                             gt::complex<T>* y)
{
  const auto gt_x = gt::adapt<1, S>(x, n);
  /* */ auto gt_y = gt::adapt<1, S>(y, n);

  gt::launch<1, S>(
    {n}, GT_LAMBDA(int j) { gt_y(j) = gt_x(j) * gt_x(j) * gt_x(j); });
}

template <typename S, typename X, typename T>
void generic_complex_op_multiply_explicit_mxp(const int n,
                                              const gt::complex<T>* x,
                                              gt::complex<T>* y)
{
  const auto mxp_x = mxp::adapt<1, S, X>(x, n);
  /* */ auto mxp_y = mxp::adapt<1, S, X>(y, n);

  gt::launch<1, S>(
    {n}, GT_LAMBDA(int j) { mxp_y(j) = mxp_x(j) * mxp_x(j) * mxp_x(j); });
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

  const gt::gtensor<complex32_t, 1> x(n, x_init);
  /* */ gt::gtensor<complex32_t, 1> y(n);

  generic_complex_op_multiply_explicit_gt<gt::space::host>(n, x.data(),
                                                           y.data());

  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, gt_ref)));

  generic_complex_op_multiply_explicit_mxp<gt::space::host, complex64_t>(
    n, x.data(), y.data());

  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, mxp_ref)));
}

TEST(mxp, complex_op_divide_implicit)
{
  using complex32_t = gt::complex<float>;
  using complex64_t = gt::complex<double>;

  const int n{2};
  double val = 1.5 + exp2f(-8) + exp2f(-15) + exp2f(-23);
  double invval = 1. / val;
  double ref = val / invval / invval;

#ifdef GTENSOR_DEVICE_CUDA
  const double lb_err_expect_gt = 1.98e-07;
#else
  const double lb_err_expect_gt = 2.7e-07;
#endif
  const double ub_err_expect_mxp = 4.0e-08;

  const complex32_t x_init{float(invval), 0.f};
  const complex32_t y_init{float(val), 0.f};

  const gt::gtensor<complex32_t, 1> x(n, x_init);
  /* */ gt::gtensor<complex32_t, 1> y_a(n, y_init);
  /* */ gt::gtensor<complex32_t, 1> y_b(n, y_init);

  const auto gt_x = gt::adapt<1>(x.data(), x.size());
  /* */ auto gt_y = gt::adapt<1>(y_a.data(), y_a.size());

  gt_y = gt_y / gt_x / gt_x;

  double gt_err = std::abs(y_a[1].real() - ref);
  EXPECT_GT(gt_err, lb_err_expect_gt);

  const auto mxp_x = mxp::adapt<1, complex64_t>(x.data(), x.size());
  /* */ auto mxp_y = mxp::adapt<1, complex64_t>(y_b.data(), y_b.size());

  mxp_y = mxp_y / mxp_x / mxp_x;

  double mxp_err = std::abs(y_b[1].real() - ref);

  EXPECT_LT(mxp_err, ub_err_expect_mxp);
}

template <typename S, typename T>
void generic_complex_op_divide_explicit_gt(const int n, const gt::complex<T>* x,
                                           gt::complex<T>* y)
{
  const auto gt_x = gt::adapt<1, S>(x, n);
  /* */ auto gt_y = gt::adapt<1, S>(y, n);

  gt::launch<1, S>(
    {n}, GT_LAMBDA(int j) { gt_y(j) = gt_y(j) / gt_x(j) / gt_x(j); });
}

template <typename S, typename X, typename T>
void generic_complex_op_divide_explicit_mxp(const int n,
                                            const gt::complex<T>* x,
                                            gt::complex<T>* y)
{
  const auto mxp_x = mxp::adapt<1, S, X>(x, n);
  /* */ auto mxp_y = mxp::adapt<1, S, X>(y, n);

  gt::launch<1, S>(
    {n}, GT_LAMBDA(int j) { mxp_y(j) = mxp_y(j) / mxp_x(j) / mxp_x(j); });
}

TEST(mxp, complex_op_divide_explicit)
{
  using complex32_t = gt::complex<float>;
  using complex64_t = gt::complex<double>;

  const int n{2};
  double val = 1.5 + exp2f(-8) + exp2f(-15) + exp2f(-23);
  double invval = 1. / val;
  double ref = val / invval / invval;

#ifdef GTENSOR_DEVICE_CUDA
  const double lb_err_expect_gt = 1.98e-07;
#else
  const double lb_err_expect_gt = 2.7e-07;
#endif
  const double ub_err_expect_mxp = 4.0e-08;

  const complex32_t x_init{float(invval), 0.f};
  const complex32_t y_init{float(val), 0.f};

  const gt::gtensor<complex32_t, 1> x(n, x_init);
  /* */ gt::gtensor<complex32_t, 1> y_a(n, y_init);
  /* */ gt::gtensor<complex32_t, 1> y_b(n, y_init);

  generic_complex_op_divide_explicit_gt<gt::space::host>(n, x.data(),
                                                         y_a.data());

  double gt_err = std::abs(y_a[1].real() - ref);
  EXPECT_GT(gt_err, lb_err_expect_gt);

  generic_complex_op_divide_explicit_mxp<gt::space::host, complex64_t>(
    n, x.data(), y_b.data());

  double mxp_err = std::abs(y_b[1].real() - ref);

  EXPECT_LT(mxp_err, ub_err_expect_mxp);
}

TEST(mxp, view_axaxaxpy)
{
  const int nx{3};
  const int ny{5};
  const float x_init{exp2f(-23)};
  const float y_init{1.f};
  const float a{1.f / 3.f};

  EXPECT_NE(y_init, y_init + x_init);

  const gt::gtensor<float, 1> x(nx, x_init);
  /* */ gt::gtensor<float, 1> y(ny, y_init);

  const auto gt_x = gt::adapt<1>(x.data(), x.size());
  /* */ auto gt_y = gt::adapt<1>(y.data(), y.size());

  using gt::placeholders::_all;
  using gt::placeholders::_s;

  gt_y.view(_s(1, -1)) = gt_y.view(_s(1, -1)) + a * gt_x.view(_all) +
                         a * gt_x.view(_all) + a * gt_x.view(_all);

  EXPECT_EQ(y, (gt::gtensor<float, 1>(ny, y_init)));

  const auto mxp_x = mxp::adapt<1, double>(x.data(), x.size());
  /* */ auto mxp_y = mxp::adapt<1, double>(y.data(), y.size());

  mxp_y.view(_s(1, -1)) = mxp_y.view(_s(1, -1)) + a * mxp_x.view(_all) +
                          a * mxp_x.view(_all) + a * mxp_x.view(_all);

  EXPECT_EQ(y, (gt::gtensor<float, 1>{y_init, y_init + x_init, y_init + x_init,
                                      y_init + x_init, y_init}));
}

TEST(mxp, view_all_2D)
{
  const int mn[2]{3, 2};
  const float x_init{exp2f(-23)};
  const float y_init{1.f};
  const float a{1.f / 3.f};

  EXPECT_NE(y_init, y_init + x_init);

  const gt::gtensor<float, 2> X(gt::shape(mn[0], mn[1]), x_init);
  /* */ gt::gtensor<float, 2> Y(gt::shape(mn[0], mn[1]), y_init);

  const auto gt_X = gt::adapt<2>(X.data(), mn);
  /* */ auto gt_Y = gt::adapt<2>(Y.data(), mn);

  using gt::placeholders::_all;

  gt_Y.view(_all, _all) = gt_Y.view(_all, _all) + a * gt_X.view(_all, _all) +
                          a * gt_X.view(_all, _all) + a * gt_X.view(_all, _all);

  EXPECT_EQ(Y, (gt::gtensor<float, 2>(gt::shape(mn[0], mn[1]), y_init)));

  const auto mxp_X = mxp::adapt<2, double>(X.data(), mn);
  /* */ auto mxp_Y = mxp::adapt<2, double>(Y.data(), mn);

  mxp_Y.view(_all, _all) = mxp_Y.view(_all, _all) + a * mxp_X.view(_all, _all) +
                           a * mxp_X.view(_all, _all) +
                           a * mxp_X.view(_all, _all);

  EXPECT_EQ(Y,
            (gt::gtensor<float, 2>(gt::shape(mn[0], mn[1]), y_init + x_init)));
}

TEST(mxp, view_newaxis_2D)
{
  const int mn[2]{3, 2};
  const float x_init{exp2f(-23)};
  const float y_init{1.f};
  const float a{1.f / 3.f};

  EXPECT_NE(y_init, y_init + x_init);

  const gt::gtensor<float, 1> x(mn[0], x_init);
  /* */ gt::gtensor<float, 2> Y(gt::shape(mn[0], mn[1]), y_init);

  const auto gt_x = gt::adapt<1>(x.data(), mn[0]);
  /* */ auto gt_Y = gt::adapt<2>(Y.data(), mn);

  using gt::placeholders::_all;
  using gt::placeholders::_newaxis;

  gt_Y.view(_all, _all) =
    gt_Y.view(_all, _all) + a * gt_x.view(_all, _newaxis) +
    a * gt_x.view(_all, _newaxis) + a * gt_x.view(_all, _newaxis);

  EXPECT_EQ(Y, (gt::gtensor<float, 2>(gt::shape(mn[0], mn[1]), y_init)));

  const auto mxp_x = mxp::adapt<1, double>(x.data(), mn[0]);
  /* */ auto mxp_Y = mxp::adapt<2, double>(Y.data(), mn);

  mxp_Y.view(_all, _all) =
    mxp_Y.view(_all, _all) + a * mxp_x.view(_all, _newaxis) +
    a * mxp_x.view(_all, _newaxis) + a * mxp_x.view(_all, _newaxis);

  EXPECT_EQ(Y,
            (gt::gtensor<float, 2>(gt::shape(mn[0], mn[1]), y_init + x_init)));
}

TEST(mxp, view_s_2D)
{
  const int mn[2]{4, 3};
  const int lj = 2, uj = 4, lk = 1, uk = 3;

  const float x_init{exp2f(-23)};
  const float y_init{1.f};
  const float a{1.f / 3.f};

  EXPECT_NE(y_init, y_init + x_init);

  const gt::gtensor<float, 2> X(gt::shape(mn[0], mn[1]), x_init);
  /* */ gt::gtensor<float, 2> Y(gt::shape(mn[0], mn[1]), y_init);

  const auto gt_X = gt::adapt<2>(X.data(), mn);
  /* */ auto gt_Y = gt::adapt<2>(Y.data(), mn);

  using gt::placeholders::_s;

  gt_Y.view(_s(lj, uj), _s(lk, uk)) = gt_Y.view(_s(lj, uj), _s(lk, uk)) +
                                      a * gt_X.view(_s(lj, uj), _s(lk, uk)) +
                                      a * gt_X.view(_s(lj, uj), _s(lk, uk)) +
                                      a * gt_X.view(_s(lj, uj), _s(lk, uk));

  EXPECT_EQ(Y, (gt::gtensor<float, 2>(gt::shape(mn[0], mn[1]), y_init)));

  const auto mxp_X = mxp::adapt<2, double>(X.data(), mn);
  /* */ auto mxp_Y = mxp::adapt<2, double>(Y.data(), mn);

  mxp_Y.view(_s(lj, uj), _s(lk, uk)) = mxp_Y.view(_s(lj, uj), _s(lk, uk)) +
                                       a * mxp_X.view(_s(lj, uj), _s(lk, uk)) +
                                       a * mxp_X.view(_s(lj, uj), _s(lk, uk)) +
                                       a * mxp_X.view(_s(lj, uj), _s(lk, uk));

  for (int j = 0; j < mn[0]; ++j) {
    for (int k = 0; k < mn[1]; ++k) {

      if (j >= lj && j < uj && k >= lk && k < uk)
        EXPECT_EQ(Y(j, k), y_init + x_init);
      else
        EXPECT_EQ(Y(j, k), y_init);
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

  const gt::gtensor<float, 2> X(gt::shape(mn[0], mn[1]), x_init);
  /* */ gt::gtensor<float, 2> Y(gt::shape(mn[0], mn[1]), y_init);

  const auto gt_X = gt::adapt<2>(X.data(), mn);
  /* */ auto gt_Y = gt::adapt<2>(Y.data(), mn);

  using gt::placeholders::_all;

  gt_Y.view(_all, slice_idx) =
    gt_Y.view(_all, slice_idx) + a * gt_X.view(_all, slice_idx) +
    a * gt_X.view(_all, slice_idx) + a * gt_X.view(_all, slice_idx);

  EXPECT_EQ(Y, (gt::gtensor<float, 2>(gt::shape(mn[0], mn[1]), y_init)));

  const auto mxp_X = mxp::adapt<2, double>(X.data(), mn);
  /* */ auto mxp_Y = mxp::adapt<2, double>(Y.data(), mn);

  mxp_Y.view(_all, slice_idx) =
    mxp_Y.view(_all, slice_idx) + a * mxp_X.view(_all, slice_idx) +
    a * mxp_X.view(_all, slice_idx) + a * mxp_X.view(_all, slice_idx);

  for (int j = 0; j < mn[0]; ++j) {
    for (int k = 0; k < mn[1]; ++k) {

      if (k == slice_idx)
        EXPECT_EQ(Y(j, k), y_init + x_init);
      else
        EXPECT_EQ(Y(j, k), y_init);
    }
  }
}

TEST(mxp, view_view_axaxaxpy)
{
  const int nx{3};
  const int ny{5};
  const float x_init{exp2f(-23)};
  const float y_init{1.f};
  const float a{1.f / 3.f};

  EXPECT_NE(y_init, y_init + x_init);

  const gt::gtensor<float, 1> x(nx, x_init);
  /* */ gt::gtensor<float, 1> y(ny, y_init);

  const auto gt_x = gt::adapt<1>(x.data(), x.size());
  /* */ auto gt_y = gt::adapt<1>(y.data(), y.size());

  using gt::placeholders::_s;

  gt_y.view(_s(1, -1)).view(_s(1, -1)) =
    gt_y.view(_s(1, -1)).view(_s(1, -1)) + a * gt_x.view(_s(1, -1)) +
    a * gt_x.view(_s(1, -1)) + a * gt_x.view(_s(1, -1));

  EXPECT_EQ(y, (gt::gtensor<float, 1>(ny, y_init)));

  const auto mxp_x = mxp::adapt<1, double>(x.data(), x.size());
  /* */ auto mxp_y = mxp::adapt<1, double>(y.data(), y.size());

  mxp_y.view(_s(1, -1)).view(_s(1, -1)) =
    mxp_y.view(_s(1, -1)).view(_s(1, -1)) + a * mxp_x.view(_s(1, -1)) +
    a * mxp_x.view(_s(1, -1)) + a * mxp_x.view(_s(1, -1));

  EXPECT_EQ(y, (gt::gtensor<float, 1>{y_init, y_init, y_init + x_init, y_init,
                                      y_init}));
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

  const gt::gtensor<float, 1> a(nx, a_init);
  const gt::gtensor<complex32_t, 1> x(nx, x_init);
  /* */ gt::gtensor<complex32_t, 1> y(ny, y_init);

  const auto gt_a = gt::adapt<1>(a.data(), a.size());
  const auto gt_x = gt::adapt<1>(x.data(), x.size());
  /* */ auto gt_y = gt::adapt<1>(y.data(), y.size());

  using gt::placeholders::_all;
  using gt::placeholders::_s;

  gt_y.view(_s(1, -1)) =
    gt_y.view(_s(1, -1)) + gt_a.view(_all) * gt_x.view(_all) +
    gt_a.view(_all) * gt_x.view(_all) + gt_a.view(_all) * gt_x.view(_all);

  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(ny, y_init)));

  const auto mxp_a = mxp::adapt<1, double>(a.data(), a.size());
  const auto mxp_x = mxp::adapt<1, complex64_t>(x.data(), x.size());
  /* */ auto mxp_y = mxp::adapt<1, complex64_t>(y.data(), y.size());

  mxp_y.view(_s(1, -1)) =
    mxp_y.view(_s(1, -1)) + mxp_a.view(_all) * mxp_x.view(_all) +
    mxp_a.view(_all) * mxp_x.view(_all) + mxp_a.view(_all) * mxp_x.view(_all);

  EXPECT_EQ(
    y, (gt::gtensor<complex32_t, 1>{y_init, y_init + x_init, y_init + x_init,
                                    y_init + x_init, y_init}));
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

  const gt::gtensor<complex32_t, 1> x(nx, x_init);
  /* */ gt::gtensor<complex32_t, 1> y(ny, y_init);

  const auto gt_x = gt::adapt<1>(x.data(), x.size());
  /* */ auto gt_y = gt::adapt<1>(y.data(), y.size());

  using gt::placeholders::_all;
  using gt::placeholders::_s;

  gt_y.view(_s(1, -1)) =
    gt_y.view(_s(1, -1)) + gt_x.view(_all) + gt_x.view(_all) + gt_x.view(_all);

  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(ny, y_init)));

  const auto mxp_x = mxp::adapt<1, complex64_t>(x.data(), x.size());
  /* */ auto mxp_y = mxp::adapt<1, complex64_t>(y.data(), y.size());

  mxp_y.view(_s(1, -1)) = mxp_y.view(_s(1, -1)) + mxp_x.view(_all) +
                          mxp_x.view(_all) + mxp_x.view(_all);

  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>{y_init, mxp_ref, mxp_ref, mxp_ref,
                                            y_init}));
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

  const gt::gtensor<complex32_t, 1> x(nx, x_init);
  /* */ gt::gtensor<complex32_t, 1> y(ny, y_init);

  const auto gt_x = gt::adapt<1>(x.data(), x.size());
  /* */ auto gt_y = gt::adapt<1>(y.data(), y.size());

  using gt::placeholders::_all;
  using gt::placeholders::_s;

  gt_y.view(_s(1, -1)) =
    gt_y.view(_s(1, -1)) - gt_x.view(_all) - gt_x.view(_all) - gt_x.view(_all);

  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(ny, y_init)));

  const auto mxp_x = mxp::adapt<1, complex64_t>(x.data(), x.size());
  /* */ auto mxp_y = mxp::adapt<1, complex64_t>(y.data(), y.size());

  mxp_y.view(_s(1, -1)) = mxp_y.view(_s(1, -1)) - mxp_x.view(_all) -
                          mxp_x.view(_all) - mxp_x.view(_all);

  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>{y_init, mxp_ref, mxp_ref, mxp_ref,
                                            y_init}));
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

  const gt::gtensor<complex32_t, 1> x(nx, x_init);
  /* */ gt::gtensor<complex32_t, 1> y(ny, y_init);

  const auto gt_x = gt::adapt<1>(x.data(), x.size());
  /* */ auto gt_y = gt::adapt<1>(y.data(), y.size());

  using gt::placeholders::_all;
  using gt::placeholders::_s;

  gt_y.view(_s(1, -1)) = gt_x.view(_all) * gt_x.view(_all) * gt_x.view(_all);

  EXPECT_EQ(
    y, (gt::gtensor<complex32_t, 1>{y_init, gt_ref, gt_ref, gt_ref, y_init}));

  const auto mxp_x = mxp::adapt<1, complex64_t>(x.data(), x.size());
  /* */ auto mxp_y = mxp::adapt<1, complex64_t>(y.data(), y.size());

  mxp_y.view(_s(1, -1)) =
    mxp_x.view(_all) * mxp_x.view(_all) * mxp_x.view(_all);

  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>{y_init, mxp_ref, mxp_ref, mxp_ref,
                                            y_init}));
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

#ifdef GTENSOR_DEVICE_CUDA
  const double lb_err_expect_gt = 1.98e-07;
#else
  const double lb_err_expect_gt = 2.7e-07;
#endif
  const double ub_err_expect_mxp = 4.0e-08;

  const complex32_t x_init{float(invval), 0.f};
  const complex32_t y_init{float(val), 0.f};

  const gt::gtensor<complex32_t, 1> x(nx, x_init);
  /* */ gt::gtensor<complex32_t, 1> y_a(ny, y_init);
  /* */ gt::gtensor<complex32_t, 1> y_b(ny, y_init);

  const auto gt_x = gt::adapt<1>(x.data(), x.size());
  /* */ auto gt_y = gt::adapt<1>(y_a.data(), y_a.size());

  using gt::placeholders::_all;
  using gt::placeholders::_s;

  gt_y.view(_s(1, -1)) =
    gt_y.view(_s(1, -1)) / gt_x.view(_all) / gt_x.view(_all);

  double gt_err = std::abs(y_a(1).real() - ref);

  EXPECT_EQ(y_a(0), y_init);
  EXPECT_GT(gt_err, lb_err_expect_gt);
  EXPECT_EQ(y_a(2), y_a(1));
  EXPECT_EQ(y_a(3), y_a(1));
  EXPECT_EQ(y_a(4), y_init);

  const auto mxp_x = mxp::adapt<1, complex64_t>(x.data(), x.size());
  /* */ auto mxp_y = mxp::adapt<1, complex64_t>(y_b.data(), y_b.size());

  mxp_y.view(_s(1, -1)) =
    mxp_y.view(_s(1, -1)) / mxp_x.view(_all) / mxp_x.view(_all);

  double mxp_err = std::abs(y_b(1).real() - ref);

  EXPECT_EQ(y_b(0), y_init);
  EXPECT_LT(mxp_err, ub_err_expect_mxp);
  EXPECT_EQ(y_b(2), y_b(1));
  EXPECT_EQ(y_b(3), y_b(1));
  EXPECT_EQ(y_b(4), y_init);
}

TEST(mxp, view_placeholders_complex_aXaXaXpY_2D)
{
  using complex32_t = gt::complex<float>;
  using complex64_t = gt::complex<double>;

  const int mn[2]{4, 3};
  const int lk = 1, uk = 2;

  const complex32_t x_init{exp2f(-23), -exp2f(-24)};
  const complex32_t y_init{1.f, 1.f};
  const float a_init{1.f / 3.f};

  EXPECT_NE(y_init.real(), y_init.real() + x_init.real());
  EXPECT_NE(y_init.imag(), y_init.imag() + x_init.imag());

  const gt::gtensor<float, 1> a(mn[0], a_init);
  const gt::gtensor<complex32_t, 1> x(mn[0], x_init);
  /* */ gt::gtensor<complex32_t, 2> Y(gt::shape(mn[0], mn[1]), y_init);

  const auto gt_a = gt::adapt<1>(a.data(), mn[0]);
  const auto gt_x = gt::adapt<1>(x.data(), mn[0]);
  /* */ auto gt_Y = gt::adapt<2>(Y.data(), mn);

  using gt::placeholders::_all;
  using gt::placeholders::_newaxis;
  using gt::placeholders::_s;

  gt_Y.view(_all, _s(lk, uk)) =
    gt_Y.view(_all, _s(lk, uk)) +
    gt_a.view(_all, _newaxis) * gt_x.view(_all, _newaxis) +
    gt_a.view(_all, _newaxis) * gt_x.view(_all, _newaxis) +
    gt_a.view(_all, _newaxis) * gt_x.view(_all, _newaxis);

  EXPECT_EQ(Y, (gt::gtensor<complex32_t, 2>(gt::shape(mn[0], mn[1]), y_init)));

  const auto mxp_a = mxp::adapt<1, double>(a.data(), mn[0]);
  const auto mxp_x = mxp::adapt<1, complex64_t>(x.data(), mn[0]);
  /* */ auto mxp_Y = mxp::adapt<2, complex64_t>(Y.data(), mn);

  mxp_Y.view(_all, _s(lk, uk)) =
    mxp_Y.view(_all, _s(lk, uk)) +
    mxp_a.view(_all, _newaxis) * mxp_x.view(_all, _newaxis) +
    mxp_a.view(_all, _newaxis) * mxp_x.view(_all, _newaxis) +
    mxp_a.view(_all, _newaxis) * mxp_x.view(_all, _newaxis);

  for (int j = 0; j < mn[0]; ++j) {
    for (int k = 0; k < mn[1]; ++k) {

      if (k >= lk && k < uk)
        EXPECT_EQ(Y(j, k), y_init + x_init);
      else
        EXPECT_EQ(Y(j, k), y_init);
    }
  }
}

#if defined GTENSOR_HAVE_DEVICE

TEST(mxp, device_axaxaxpy_implicit)
{
  const int n{2};
  const float x_init{exp2f(-23)};
  const float y_init{1.f};
  const float a{1.f / 3.f};

  EXPECT_NE(y_init, y_init + x_init);

  const gt::gtensor_device<float, 1> x(n, x_init);
  /* */ gt::gtensor_device<float, 1> y(n, y_init);

  const auto gt_x = gt::adapt<1, gt::space::device>(x.data(), x.size());
  /* */ auto gt_y = gt::adapt<1, gt::space::device>(y.data(), y.size());

  gt_y = gt_y + a * gt_x + a * gt_x + a * gt_x;

  EXPECT_EQ(y, (gt::gtensor_device<float, 1>(n, y_init)));

  const auto mxp_x =
    mxp::adapt<1, gt::space::device, double>(x.data(), x.size());
  /* */ auto mxp_y =
    mxp::adapt<1, gt::space::device, double>(y.data(), y.size());

  mxp_y = mxp_y + a * mxp_x + a * mxp_x + a * mxp_x;

  EXPECT_EQ(y, (gt::gtensor_device<float, 1>(n, y_init + x_init)));
}

TEST(mxp, device_axaxaxpy_explicit)
{
  const int n{2};
  const float x_init{exp2f(-23)};
  const float y_init{1.f};
  const float a{1.f / 3.f};

  EXPECT_NE(y_init, y_init + x_init);

  const gt::gtensor_device<float, 1> x(n, x_init);
  /* */ gt::gtensor_device<float, 1> y(n, y_init);

  generic_axaxaxpy_explicit_gt<gt::space::device>(n, a, x, y);

  EXPECT_EQ(y, (gt::gtensor<float, 1>(n, y_init)));

  generic_axaxaxpy_explicit_mxp<gt::space::device, double>(n, a, x, y);

  EXPECT_EQ(y, (gt::gtensor<float, 1>(n, y_init + x_init)));
}

TEST(mxp, device_aXaXaXpY_2D_implicit)
{
  const int mn[2]{2, 3};
  const float x_init{exp2f(-23)};
  const float y_init{1.f};
  const float a{1.f / 3.f};

  EXPECT_NE(y_init, y_init + x_init);

  const gt::gtensor_device<float, 2> X(gt::shape(mn[0], mn[1]), x_init);
  /* */ gt::gtensor_device<float, 2> Y(gt::shape(mn[0], mn[1]), y_init);

  const auto gt_X = gt::adapt_device<2>(gt::raw_pointer_cast(X.data()), mn);
  /* */ auto gt_Y = gt::adapt_device<2>(gt::raw_pointer_cast(Y.data()), mn);

  gt_Y = gt_Y + a * gt_X + a * gt_X + a * gt_X;

  EXPECT_EQ(Y, (gt::gtensor_device<float, 2>(gt::shape(mn[0], mn[1]), y_init)));

  const auto mxp_X =
    mxp::adapt_device<2, double>(gt::raw_pointer_cast(X.data()), mn);
  /* */ auto mxp_Y =
    mxp::adapt_device<2, double>(gt::raw_pointer_cast(Y.data()), mn);

  mxp_Y = mxp_Y + a * mxp_X + a * mxp_X + a * mxp_X;

  EXPECT_EQ(Y, (gt::gtensor_device<float, 2>(gt::shape(mn[0], mn[1]),
                                             y_init + x_init)));
}

TEST(mxp, device_aXaXaXpY_2D_explicit)
{
  const int mn[2]{2, 3};
  const float x_init{exp2f(-23)};
  const float y_init{1.f};
  const float a{1.f / 3.f};

  EXPECT_NE(y_init, y_init + x_init);

  const gt::gtensor_device<float, 2> X(gt::shape(mn[0], mn[1]), x_init);
  /* */ gt::gtensor_device<float, 2> Y(gt::shape(mn[0], mn[1]), y_init);

  generic_aXaXaXpy_2D_explicit_gt<gt::space::device>(mn, a, X, Y);

  EXPECT_EQ(Y, (gt::gtensor<float, 2>(gt::shape(mn[0], mn[1]), y_init)));

  generic_aXaXaXpy_2D_explicit_mxp<gt::space::device, double>(mn, a, X, Y);

  EXPECT_EQ(Y,
            (gt::gtensor<float, 2>(gt::shape(mn[0], mn[1]), y_init + x_init)));
}

#endif
