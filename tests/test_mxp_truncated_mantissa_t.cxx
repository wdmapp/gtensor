#include <gtest/gtest.h>

#include <gtensor/gtensor.h>
#include <gtensor/reductions.h>

#include <gtensor/mxp.h>

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <type_traits>

// -------------------------------------------------------------------------- //

template <std::uint8_t From, std::uint8_t To, typename Task>
struct Loop
{
  template <typename... Args>
  static std::enable_if_t<From <= To> Run(Args&&... args)
  {
    Task::template Iteration<From>(std::forward<Args>(args)...);
    Loop<From + 1, To, Task>::Run(args...);
  }
};

template <std::uint8_t FromTo, typename Task>
struct Loop<FromTo, FromTo, Task>
{
  template <typename... Args>
  static void Run(Args&&... args)
  {
    Task::template Iteration<FromTo>(std::forward<Args>(args)...);
  }
};

// -------------------------------------------------------------------------- //

template <std::uint8_t bits, typename S, typename T>
void generic_truncated_add(const gt::gtensor<T, 1, S>& x,
                           gt::gtensor<T, 1, S>& y)
{
  using mxp_type = mxp::truncated_mantissa_t<T, bits>;
  const auto mxp_x = mxp::adapt<1, S, mxp_type>(x.data(), x.size());
  auto mxp_y = mxp::adapt<1, S, mxp_type>(y.data(), y.size());

  mxp_y = mxp_y + mxp_x;
}

template <std::uint8_t bits>
float ref_truncated_add_float()
{
  if (bits < 12)
    return 2.f;
  else if (bits == 12)
    return 2.f + exp2f(-12);
  else if (bits == 13)
    return 2.f + exp2f(-13);
  else if (bits == 14)
    return 2.f + exp2f(-13) + exp2f(-14);
  else if (bits == 15)
    return 2.f + exp2f(-13) + exp2f(-15);
  else // bits > 15
    return 2.f + exp2f(-13) + exp2f(-15) + exp2f(-16);
}

template <std::uint8_t bits>
double ref_truncated_add_double()
{
  if (bits < 22)
    return 2.;
  else if (bits == 22)
    return 2. + exp2(-22);
  else if (bits == 23)
    return 2. + exp2(-23);
  else if (bits == 24)
    return 2. + exp2(-23) + exp2(-24);
  else if (bits == 25)
    return 2. + exp2(-23) + exp2(-25);
  else // bits > 25
    return 2. + exp2(-23) + exp2(-25) + exp2(-26);
}

template <std::uint8_t bits, typename T>
T ref_truncated_add_gen()
{
  if (std::is_same<T, float>::value ||
      std::is_same<T, gt::complex<float>>::value)
    return ref_truncated_add_float<bits>();
  else if (std::is_same<T, double>::value ||
           std::is_same<T, gt::complex<double>>::value)
    return ref_truncated_add_double<bits>();
  else
    return 0. / 0.;
}

struct run_test_add_host
{
  using S = gt::space::host;

  template <std::uint8_t bits, typename T>
  static void Iteration(const gt::gtensor<T, 1, S>& x, gt::gtensor<T, 1, S>& y,
                        const T y_init)
  {
    auto gt_y = gt::adapt<1, S>(y.data(), y.size());
    y.view() = y_init;

    generic_truncated_add<bits, S>(x, y);
    EXPECT_EQ(
      y, (gt::gtensor<T, 1, S>(y.size(), ref_truncated_add_gen<bits, T>())));
  }
};

TEST(mxp_truncated_mantissa, add_float)
{
  const int n{3};

  const float x_init{1.f + exp2f(-13) + exp2f(-15) + exp2f(-16)};
  const float y_init{1.f};

  EXPECT_NE(y_init, y_init + x_init);

  const gt::gtensor<float, 1> x(n, x_init);
  gt::gtensor<float, 1> y(n, y_init);

  Loop<0, 23, run_test_add_host>::Run(x, y, y_init);
}

TEST(mxp_truncated_mantissa, add_double)
{
  const int n{3};

  const double x_init{1. + exp2(-23) + exp2(-25) + exp2(-26)};
  const double y_init{1.};

  EXPECT_NE(y_init, y_init + x_init);

  const gt::gtensor<double, 1> x(n, x_init);
  gt::gtensor<double, 1> y(n, y_init);

  Loop<0, 52, run_test_add_host>::Run(x, y, y_init);
}

TEST(mxp_truncated_mantissa, add_complex_float)
{
  using complex32_t = gt::complex<float>;

  const int n{3};

  const complex32_t x_init{1.f + exp2f(-13) + exp2f(-15) + exp2f(-16)};
  const complex32_t y_init{1.f};

  EXPECT_NE(y_init, y_init + x_init);

  const gt::gtensor<complex32_t, 1> x(n, x_init);
  gt::gtensor<complex32_t, 1> y(n, y_init);

  Loop<0, 23, run_test_add_host>::Run(x, y, y_init);
}

TEST(mxp_truncated_mantissa, add_complex_double)
{
  using complex64_t = gt::complex<double>;

  const int n{3};

  const complex64_t x_init{1. + exp2(-23) + exp2(-25) + exp2(-26)};
  const complex64_t y_init{1.};

  EXPECT_NE(y_init, y_init + x_init);

  const gt::gtensor<complex64_t, 1> x(n, x_init);
  gt::gtensor<complex64_t, 1> y(n, y_init);

  Loop<0, 52, run_test_add_host>::Run(x, y, y_init);
}

template <std::uint8_t bits, typename S, typename T>
void generic_view_truncated_add(const gt::gtensor<T, 1, S>& x,
                                gt::gtensor<T, 1, S>& y)
{
  using mxp_type = mxp::truncated_mantissa_t<T, bits>;
  const auto mxp_x = mxp::adapt<1, S, mxp_type>(x.data(), x.size());
  auto mxp_y = mxp::adapt<1, S, mxp_type>(y.data(), y.size());

  using gt::placeholders::_all;
  using gt::placeholders::_s;

  mxp_y.view(_s(1, -1)) = mxp_y.view(_s(1, -1)) + mxp_x.view(_all);
}

struct run_test_view_add_host
{
  using S = gt::space::host;

  template <std::uint8_t bits, typename T>
  static void Iteration(const gt::gtensor<T, 1, S>& x, gt::gtensor<T, 1, S>& y,
                        const T y_init)
  {
    auto gt_y = gt::adapt<1, S>(y.data(), y.size());
    y.view() = y_init;

    generic_view_truncated_add<bits, S>(x, y);
    EXPECT_EQ(y,
              (gt::gtensor<T, 1, S>{y_init, ref_truncated_add_gen<bits, T>(),
                                    ref_truncated_add_gen<bits, T>(),
                                    ref_truncated_add_gen<bits, T>(), y_init}));
  }
};

TEST(mxp_truncated_mantissa, view_add_float)
{
  const int nx{3};
  const int ny{5};

  const float x_init{1.f + exp2f(-13) + exp2f(-15) + exp2f(-16)};
  const float y_init{1.f};

  EXPECT_NE(y_init, y_init + x_init);

  const gt::gtensor<float, 1> x(nx, x_init);
  gt::gtensor<float, 1> y(ny, y_init);

  Loop<0, 23, run_test_view_add_host>::Run(x, y, y_init);
}

TEST(mxp_truncated_mantissa, view_add_double)
{
  const int nx{3};
  const int ny{5};

  const double x_init{1. + exp2(-23) + exp2(-25) + exp2(-26)};
  const double y_init{1.};

  EXPECT_NE(y_init, y_init + x_init);

  const gt::gtensor<double, 1> x(nx, x_init);
  gt::gtensor<double, 1> y(ny, y_init);

  Loop<0, 52, run_test_view_add_host>::Run(x, y, y_init);
}

TEST(mxp_truncated_mantissa, view_add_complex_float)
{
  using complex32_t = gt::complex<float>;

  const int nx{3};
  const int ny{5};

  const complex32_t x_init{1.f + exp2f(-13) + exp2f(-15) + exp2f(-16)};
  const complex32_t y_init{1.f};

  EXPECT_NE(y_init, y_init + x_init);

  const gt::gtensor<complex32_t, 1> x(nx, x_init);
  gt::gtensor<complex32_t, 1> y(ny, y_init);

  Loop<0, 23, run_test_view_add_host>::Run(x, y, y_init);
}

TEST(mxp_truncated_mantissa, view_add_complex_double)
{
  using complex64_t = gt::complex<double>;

  const int nx{3};
  const int ny{5};

  const complex64_t x_init{1. + exp2(-23) + exp2(-25) + exp2(-26)};
  const complex64_t y_init{1.};

  EXPECT_NE(y_init, y_init + x_init);

  const gt::gtensor<complex64_t, 1> x(nx, x_init);
  gt::gtensor<complex64_t, 1> y(ny, y_init);
  auto gt_y = gt::adapt<1>(y.data(), y.size());

  gt_y.view() = y_init;

  Loop<0, 52, run_test_view_add_host>::Run(x, y, y_init);
}

template <std::uint8_t bits, typename S, typename T>
void generic_view_2D_truncated_add(const gt::gtensor<T, 1, S>& x,
                                   gt::gtensor<T, 2, S>& y)
{
  using mxp_type = mxp::truncated_mantissa_t<T, bits>;
  const auto mxp_x = mxp::adapt<1, S, mxp_type>(x.data(), x.size());
  auto mxp_y = mxp::adapt<2, S, mxp_type>(y.data(), y.shape());

  using gt::placeholders::_all;
  using gt::placeholders::_s;

  mxp_y.view(_s(1, -1), 1) = mxp_y.view(_s(1, -1), 1) + mxp_x.view(_all);
}

struct run_test_view_2D_add_host
{
  using S = gt::space::host;

  template <std::uint8_t bits, typename T>
  static void Iteration(const gt::gtensor<T, 1, S>& x, gt::gtensor<T, 2, S>& y,
                        const T y_init)
  {
    auto gt_y = gt::adapt<2, S>(y.data(), y.shape());
    y.view() = y_init;

    generic_view_2D_truncated_add<bits, S>(x, y);
    EXPECT_EQ(
      y, (gt::gtensor<T, 2, S>{{y_init, y_init, y_init, y_init, y_init},
                               {y_init, ref_truncated_add_gen<bits, T>(),
                                ref_truncated_add_gen<bits, T>(),
                                ref_truncated_add_gen<bits, T>(), y_init}}));
  }
};

TEST(mxp_truncated_mantissa, view_2D_add_float)
{
  const int nx{3};
  const int mny[2]{5, 2};

  const float x_init{1.f + exp2f(-13) + exp2f(-15) + exp2f(-16)};
  const float y_init{1.f};

  EXPECT_NE(y_init, y_init + x_init);

  const gt::gtensor<float, 1> x(nx, x_init);
  gt::gtensor<float, 2> y(gt::shape(mny[0], mny[1]), y_init);

  Loop<0, 23, run_test_view_2D_add_host>::Run(x, y, y_init);
}

TEST(mxp_truncated_mantissa, view_2D_add_double)
{
  const int nx{3};
  const int mny[2]{5, 2};

  const double x_init{1. + exp2(-23) + exp2(-25) + exp2(-26)};
  const double y_init{1.};

  EXPECT_NE(y_init, y_init + x_init);

  const gt::gtensor<double, 1> x(nx, x_init);
  gt::gtensor<double, 2> y(gt::shape(mny[0], mny[1]), y_init);

  Loop<0, 52, run_test_view_2D_add_host>::Run(x, y, y_init);
}

TEST(mxp_truncated_mantissa, view_2D_add_complex_float)
{
  using complex32_t = gt::complex<float>;

  const int nx{3};
  const int mny[2]{5, 2};

  const complex32_t x_init{1.f + exp2f(-13) + exp2f(-15) + exp2f(-16)};
  const complex32_t y_init{1.f};

  EXPECT_NE(y_init, y_init + x_init);

  const gt::gtensor<complex32_t, 1> x(nx, x_init);
  gt::gtensor<complex32_t, 2> y(gt::shape(mny[0], mny[1]), y_init);

  Loop<0, 23, run_test_view_2D_add_host>::Run(x, y, y_init);
}

TEST(mxp_truncated_mantissa, view_2D_add_complex_double)
{
  using complex64_t = gt::complex<double>;

  const int nx{3};
  const int mny[2]{5, 2};

  const complex64_t x_init{1. + exp2(-23) + exp2(-25) + exp2(-26)};
  const complex64_t y_init{1.};

  EXPECT_NE(y_init, y_init + x_init);

  const gt::gtensor<complex64_t, 1> x(nx, x_init);
  gt::gtensor<complex64_t, 2> y(gt::shape(mny[0], mny[1]), y_init);

  Loop<0, 52, run_test_view_2D_add_host>::Run(x, y, y_init);
}

struct run_test_error_bounds_host
{
  using S = gt::space::host;

  template <std::uint8_t bits, typename T>
  static void Iteration(const gt::gtensor<T, 1, S>& x, gt::gtensor<T, 1, S>& y)
  {
    const T hard_threshold = std::pow(T{2.}, -(bits + 1));

    using mxp_type = mxp::truncated_mantissa_t<T, bits>;
    const auto mxp_x = mxp::adapt<1, S, mxp_type>(x.data(), x.size());
    const auto gt_x = gt::adapt<1, S>(x.data(), x.size());
    /* */ auto gt_y = gt::adapt<1, S>(y.data(), y.size());

    gt_y = gt_x - mxp_x;

    auto error = gt::norm_linf(y);
    EXPECT_LE(error, hard_threshold);
  }
};

TEST(mxp_truncated_mantissa, error_bounds_float)
{
  const int n{1000};

  gt::gtensor<float, 1> x(n);
  gt::gtensor<float, 1> y(n);

  std::srand(time(nullptr));
  for (int j = 0; j < x.size(); ++j)
    x(j) = 1.f + 1.f * std::rand() /
                   std::numeric_limits<decltype(std::rand())>::max();

  Loop<0, 23, run_test_error_bounds_host>::Run(x, y);
}

TEST(mxp_truncated_mantissa, error_bounds_double)
{
  const int n{1000};

  gt::gtensor<double, 1> x(n);
  gt::gtensor<double, 1> y(n);

  std::srand(time(nullptr));
  for (int j = 0; j < x.size(); ++j)
    x(j) =
      1. + 1. * std::rand() / std::numeric_limits<decltype(std::rand())>::max();

  Loop<0, 52, run_test_error_bounds_host>::Run(x, y);
}
