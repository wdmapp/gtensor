#include <gtest/gtest.h>

#include <gtensor/gtensor.h>

#include <gtensor/mxp.h>

#include <cmath>

template <std::uint8_t bits, typename S, typename T>
void generic_truncated_add(const gt::gtensor<T, 1, S>& x,
                           gt::gtensor<T, 1, S>& y)
{
  using mxp_type = mxp::truncated_mantissa_t<T, bits>;
  const auto mxp_x = mxp::adapt<1, S, mxp_type>(x.data(), x.size());
  /* */ auto mxp_y = mxp::adapt<1, S, mxp_type>(y.data(), y.size());

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

template <std::uint8_t bits, typename S, typename T>
void run_test_add(const gt::gtensor<T, 1, S>& x, gt::gtensor<T, 1, S>& y,
                  const T y_init)
{
  auto gt_y = gt::adapt<1, S>(y.data(), y.size());
  y.view() = y_init;

  generic_truncated_add<bits, S>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<T, 1, S>(y.size(), ref_truncated_add_gen<bits, T>())));
}

TEST(mxp_truncated_mantissa, add_float)
{
  const int n{3};

  const float x_init{1.f + exp2f(-13) + exp2f(-15) + exp2f(-16)};
  const float y_init{1.f};

  EXPECT_NE(y_init, y_init + x_init);

  const gt::gtensor<float, 1> x(n, x_init);
  /* */ gt::gtensor<float, 1> y(n, y_init);

  run_test_add<0, gt::space::host>(x, y, y_init);
  run_test_add<1, gt::space::host>(x, y, y_init);
  run_test_add<2, gt::space::host>(x, y, y_init);
  run_test_add<3, gt::space::host>(x, y, y_init);
  run_test_add<4, gt::space::host>(x, y, y_init);
  run_test_add<5, gt::space::host>(x, y, y_init);
  run_test_add<6, gt::space::host>(x, y, y_init);
  run_test_add<7, gt::space::host>(x, y, y_init);
  run_test_add<8, gt::space::host>(x, y, y_init);
  run_test_add<9, gt::space::host>(x, y, y_init);
  run_test_add<10, gt::space::host>(x, y, y_init);
  run_test_add<11, gt::space::host>(x, y, y_init);
  run_test_add<12, gt::space::host>(x, y, y_init);
  run_test_add<13, gt::space::host>(x, y, y_init);
  run_test_add<14, gt::space::host>(x, y, y_init);
  run_test_add<15, gt::space::host>(x, y, y_init);
  run_test_add<16, gt::space::host>(x, y, y_init);
  run_test_add<17, gt::space::host>(x, y, y_init);
  run_test_add<18, gt::space::host>(x, y, y_init);
  run_test_add<19, gt::space::host>(x, y, y_init);
  run_test_add<20, gt::space::host>(x, y, y_init);
  run_test_add<21, gt::space::host>(x, y, y_init);
  run_test_add<22, gt::space::host>(x, y, y_init);
  run_test_add<23, gt::space::host>(x, y, y_init);
}

TEST(mxp_truncated_mantissa, add_double)
{
  const int n{3};

  const double x_init{1. + exp2(-23) + exp2(-25) + exp2(-26)};
  const double y_init{1.};

  EXPECT_NE(y_init, y_init + x_init);

  const gt::gtensor<double, 1> x(n, x_init);
  /* */ gt::gtensor<double, 1> y(n, y_init);

  run_test_add<0, gt::space::host>(x, y, y_init);
  run_test_add<1, gt::space::host>(x, y, y_init);
  run_test_add<2, gt::space::host>(x, y, y_init);
  run_test_add<3, gt::space::host>(x, y, y_init);
  run_test_add<4, gt::space::host>(x, y, y_init);
  run_test_add<5, gt::space::host>(x, y, y_init);
  run_test_add<6, gt::space::host>(x, y, y_init);
  run_test_add<7, gt::space::host>(x, y, y_init);
  run_test_add<8, gt::space::host>(x, y, y_init);
  run_test_add<9, gt::space::host>(x, y, y_init);
  run_test_add<10, gt::space::host>(x, y, y_init);
  run_test_add<11, gt::space::host>(x, y, y_init);
  run_test_add<12, gt::space::host>(x, y, y_init);
  run_test_add<13, gt::space::host>(x, y, y_init);
  run_test_add<14, gt::space::host>(x, y, y_init);
  run_test_add<15, gt::space::host>(x, y, y_init);
  run_test_add<16, gt::space::host>(x, y, y_init);
  run_test_add<17, gt::space::host>(x, y, y_init);
  run_test_add<18, gt::space::host>(x, y, y_init);
  run_test_add<19, gt::space::host>(x, y, y_init);
  run_test_add<20, gt::space::host>(x, y, y_init);
  run_test_add<21, gt::space::host>(x, y, y_init);
  run_test_add<22, gt::space::host>(x, y, y_init);
  run_test_add<23, gt::space::host>(x, y, y_init);
  run_test_add<24, gt::space::host>(x, y, y_init);
  run_test_add<25, gt::space::host>(x, y, y_init);
  run_test_add<26, gt::space::host>(x, y, y_init);
  run_test_add<27, gt::space::host>(x, y, y_init);
  run_test_add<28, gt::space::host>(x, y, y_init);
  run_test_add<29, gt::space::host>(x, y, y_init);
  run_test_add<30, gt::space::host>(x, y, y_init);
  run_test_add<31, gt::space::host>(x, y, y_init);
  run_test_add<32, gt::space::host>(x, y, y_init);
  run_test_add<33, gt::space::host>(x, y, y_init);
  run_test_add<34, gt::space::host>(x, y, y_init);
  run_test_add<35, gt::space::host>(x, y, y_init);
  run_test_add<36, gt::space::host>(x, y, y_init);
  run_test_add<37, gt::space::host>(x, y, y_init);
  run_test_add<38, gt::space::host>(x, y, y_init);
  run_test_add<39, gt::space::host>(x, y, y_init);
  run_test_add<40, gt::space::host>(x, y, y_init);
  run_test_add<41, gt::space::host>(x, y, y_init);
  run_test_add<42, gt::space::host>(x, y, y_init);
  run_test_add<43, gt::space::host>(x, y, y_init);
  run_test_add<44, gt::space::host>(x, y, y_init);
  run_test_add<45, gt::space::host>(x, y, y_init);
  run_test_add<46, gt::space::host>(x, y, y_init);
  run_test_add<47, gt::space::host>(x, y, y_init);
  run_test_add<48, gt::space::host>(x, y, y_init);
  run_test_add<49, gt::space::host>(x, y, y_init);
  run_test_add<50, gt::space::host>(x, y, y_init);
  run_test_add<51, gt::space::host>(x, y, y_init);
  run_test_add<52, gt::space::host>(x, y, y_init);
  run_test_add<53, gt::space::host>(x, y, y_init);
}

TEST(mxp_truncated_mantissa, add_complex_float)
{
  using complex32_t = gt::complex<float>;

  const int n{3};

  const complex32_t x_init{1.f + exp2f(-13) + exp2f(-15) + exp2f(-16)};
  const complex32_t y_init{1.f};

  EXPECT_NE(y_init, y_init + x_init);

  const gt::gtensor<complex32_t, 1> x(n, x_init);
  /* */ gt::gtensor<complex32_t, 1> y(n, y_init);

  run_test_add<0, gt::space::host>(x, y, y_init);
  run_test_add<1, gt::space::host>(x, y, y_init);
  run_test_add<2, gt::space::host>(x, y, y_init);
  run_test_add<3, gt::space::host>(x, y, y_init);
  run_test_add<4, gt::space::host>(x, y, y_init);
  run_test_add<5, gt::space::host>(x, y, y_init);
  run_test_add<6, gt::space::host>(x, y, y_init);
  run_test_add<7, gt::space::host>(x, y, y_init);
  run_test_add<8, gt::space::host>(x, y, y_init);
  run_test_add<9, gt::space::host>(x, y, y_init);
  run_test_add<10, gt::space::host>(x, y, y_init);
  run_test_add<11, gt::space::host>(x, y, y_init);
  run_test_add<12, gt::space::host>(x, y, y_init);
  run_test_add<13, gt::space::host>(x, y, y_init);
  run_test_add<14, gt::space::host>(x, y, y_init);
  run_test_add<15, gt::space::host>(x, y, y_init);
  run_test_add<16, gt::space::host>(x, y, y_init);
  run_test_add<17, gt::space::host>(x, y, y_init);
  run_test_add<18, gt::space::host>(x, y, y_init);
  run_test_add<19, gt::space::host>(x, y, y_init);
  run_test_add<20, gt::space::host>(x, y, y_init);
  run_test_add<21, gt::space::host>(x, y, y_init);
  run_test_add<22, gt::space::host>(x, y, y_init);
  run_test_add<23, gt::space::host>(x, y, y_init);
}

TEST(mxp_truncated_mantissa, add_complex_double)
{
  using complex64_t = gt::complex<double>;

  const int n{3};

  const complex64_t x_init{1. + exp2(-23) + exp2(-25) + exp2(-26)};
  const complex64_t y_init{1.};

  EXPECT_NE(y_init, y_init + x_init);

  const gt::gtensor<complex64_t, 1> x(n, x_init);
  /* */ gt::gtensor<complex64_t, 1> y(n, y_init);

  run_test_add<0, gt::space::host>(x, y, y_init);
  run_test_add<1, gt::space::host>(x, y, y_init);
  run_test_add<2, gt::space::host>(x, y, y_init);
  run_test_add<3, gt::space::host>(x, y, y_init);
  run_test_add<4, gt::space::host>(x, y, y_init);
  run_test_add<5, gt::space::host>(x, y, y_init);
  run_test_add<6, gt::space::host>(x, y, y_init);
  run_test_add<7, gt::space::host>(x, y, y_init);
  run_test_add<8, gt::space::host>(x, y, y_init);
  run_test_add<9, gt::space::host>(x, y, y_init);
  run_test_add<10, gt::space::host>(x, y, y_init);
  run_test_add<11, gt::space::host>(x, y, y_init);
  run_test_add<12, gt::space::host>(x, y, y_init);
  run_test_add<13, gt::space::host>(x, y, y_init);
  run_test_add<14, gt::space::host>(x, y, y_init);
  run_test_add<15, gt::space::host>(x, y, y_init);
  run_test_add<16, gt::space::host>(x, y, y_init);
  run_test_add<17, gt::space::host>(x, y, y_init);
  run_test_add<18, gt::space::host>(x, y, y_init);
  run_test_add<19, gt::space::host>(x, y, y_init);
  run_test_add<20, gt::space::host>(x, y, y_init);
  run_test_add<21, gt::space::host>(x, y, y_init);
  run_test_add<22, gt::space::host>(x, y, y_init);
  run_test_add<23, gt::space::host>(x, y, y_init);
  run_test_add<24, gt::space::host>(x, y, y_init);
  run_test_add<25, gt::space::host>(x, y, y_init);
  run_test_add<26, gt::space::host>(x, y, y_init);
  run_test_add<27, gt::space::host>(x, y, y_init);
  run_test_add<28, gt::space::host>(x, y, y_init);
  run_test_add<29, gt::space::host>(x, y, y_init);
  run_test_add<30, gt::space::host>(x, y, y_init);
  run_test_add<31, gt::space::host>(x, y, y_init);
  run_test_add<32, gt::space::host>(x, y, y_init);
  run_test_add<33, gt::space::host>(x, y, y_init);
  run_test_add<34, gt::space::host>(x, y, y_init);
  run_test_add<35, gt::space::host>(x, y, y_init);
  run_test_add<36, gt::space::host>(x, y, y_init);
  run_test_add<37, gt::space::host>(x, y, y_init);
  run_test_add<38, gt::space::host>(x, y, y_init);
  run_test_add<39, gt::space::host>(x, y, y_init);
  run_test_add<40, gt::space::host>(x, y, y_init);
  run_test_add<41, gt::space::host>(x, y, y_init);
  run_test_add<42, gt::space::host>(x, y, y_init);
  run_test_add<43, gt::space::host>(x, y, y_init);
  run_test_add<44, gt::space::host>(x, y, y_init);
  run_test_add<45, gt::space::host>(x, y, y_init);
  run_test_add<46, gt::space::host>(x, y, y_init);
  run_test_add<47, gt::space::host>(x, y, y_init);
  run_test_add<48, gt::space::host>(x, y, y_init);
  run_test_add<49, gt::space::host>(x, y, y_init);
  run_test_add<50, gt::space::host>(x, y, y_init);
  run_test_add<51, gt::space::host>(x, y, y_init);
  run_test_add<52, gt::space::host>(x, y, y_init);
  run_test_add<53, gt::space::host>(x, y, y_init);
}

template <std::uint8_t bits, typename S, typename T>
void generic_view_truncated_add(const gt::gtensor<T, 1, S>& x,
                                gt::gtensor<T, 1, S>& y)
{
  using mxp_type = mxp::truncated_mantissa_t<T, bits>;
  const auto mxp_x = mxp::adapt<1, S, mxp_type>(x.data(), x.size());
  /* */ auto mxp_y = mxp::adapt<1, S, mxp_type>(y.data(), y.size());

  using gt::placeholders::_all;
  using gt::placeholders::_s;

  mxp_y.view(_s(1, -1)) = mxp_y.view(_s(1, -1)) + mxp_x.view(_all);
}

template <std::uint8_t bits, typename S, typename T>
void run_test_view_add(const gt::gtensor<T, 1, S>& x, gt::gtensor<T, 1, S>& y,
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

TEST(mxp_truncated_mantissa, view_add_float)
{
  const int nx{3};
  const int ny{5};

  const float x_init{1.f + exp2f(-13) + exp2f(-15) + exp2f(-16)};
  const float y_init{1.f};

  EXPECT_NE(y_init, y_init + x_init);

  const gt::gtensor<float, 1> x(nx, x_init);
  /* */ gt::gtensor<float, 1> y(ny, y_init);

  run_test_view_add<0, gt::space::host>(x, y, y_init);
  run_test_view_add<1, gt::space::host>(x, y, y_init);
  run_test_view_add<2, gt::space::host>(x, y, y_init);
  run_test_view_add<3, gt::space::host>(x, y, y_init);
  run_test_view_add<4, gt::space::host>(x, y, y_init);
  run_test_view_add<5, gt::space::host>(x, y, y_init);
  run_test_view_add<6, gt::space::host>(x, y, y_init);
  run_test_view_add<7, gt::space::host>(x, y, y_init);
  run_test_view_add<8, gt::space::host>(x, y, y_init);
  run_test_view_add<9, gt::space::host>(x, y, y_init);
  run_test_view_add<10, gt::space::host>(x, y, y_init);
  run_test_view_add<11, gt::space::host>(x, y, y_init);
  run_test_view_add<12, gt::space::host>(x, y, y_init);
  run_test_view_add<13, gt::space::host>(x, y, y_init);
  run_test_view_add<14, gt::space::host>(x, y, y_init);
  run_test_view_add<15, gt::space::host>(x, y, y_init);
  run_test_view_add<16, gt::space::host>(x, y, y_init);
  run_test_view_add<17, gt::space::host>(x, y, y_init);
  run_test_view_add<18, gt::space::host>(x, y, y_init);
  run_test_view_add<19, gt::space::host>(x, y, y_init);
  run_test_view_add<20, gt::space::host>(x, y, y_init);
  run_test_view_add<21, gt::space::host>(x, y, y_init);
  run_test_view_add<22, gt::space::host>(x, y, y_init);
  run_test_view_add<23, gt::space::host>(x, y, y_init);
}

TEST(mxp_truncated_mantissa, view_add_double)
{
  const int nx{3};
  const int ny{5};

  const double x_init{1. + exp2(-23) + exp2(-25) + exp2(-26)};
  const double y_init{1.};

  EXPECT_NE(y_init, y_init + x_init);

  const gt::gtensor<double, 1> x(nx, x_init);
  /* */ gt::gtensor<double, 1> y(ny, y_init);

  run_test_view_add<0, gt::space::host>(x, y, y_init);
  run_test_view_add<1, gt::space::host>(x, y, y_init);
  run_test_view_add<2, gt::space::host>(x, y, y_init);
  run_test_view_add<3, gt::space::host>(x, y, y_init);
  run_test_view_add<4, gt::space::host>(x, y, y_init);
  run_test_view_add<5, gt::space::host>(x, y, y_init);
  run_test_view_add<6, gt::space::host>(x, y, y_init);
  run_test_view_add<7, gt::space::host>(x, y, y_init);
  run_test_view_add<8, gt::space::host>(x, y, y_init);
  run_test_view_add<9, gt::space::host>(x, y, y_init);
  run_test_view_add<10, gt::space::host>(x, y, y_init);
  run_test_view_add<11, gt::space::host>(x, y, y_init);
  run_test_view_add<12, gt::space::host>(x, y, y_init);
  run_test_view_add<13, gt::space::host>(x, y, y_init);
  run_test_view_add<14, gt::space::host>(x, y, y_init);
  run_test_view_add<15, gt::space::host>(x, y, y_init);
  run_test_view_add<16, gt::space::host>(x, y, y_init);
  run_test_view_add<17, gt::space::host>(x, y, y_init);
  run_test_view_add<18, gt::space::host>(x, y, y_init);
  run_test_view_add<19, gt::space::host>(x, y, y_init);
  run_test_view_add<20, gt::space::host>(x, y, y_init);
  run_test_view_add<21, gt::space::host>(x, y, y_init);
  run_test_view_add<22, gt::space::host>(x, y, y_init);
  run_test_view_add<23, gt::space::host>(x, y, y_init);
  run_test_view_add<24, gt::space::host>(x, y, y_init);
  run_test_view_add<25, gt::space::host>(x, y, y_init);
  run_test_view_add<26, gt::space::host>(x, y, y_init);
  run_test_view_add<27, gt::space::host>(x, y, y_init);
  run_test_view_add<28, gt::space::host>(x, y, y_init);
  run_test_view_add<29, gt::space::host>(x, y, y_init);
  run_test_view_add<30, gt::space::host>(x, y, y_init);
  run_test_view_add<31, gt::space::host>(x, y, y_init);
  run_test_view_add<32, gt::space::host>(x, y, y_init);
  run_test_view_add<33, gt::space::host>(x, y, y_init);
  run_test_view_add<34, gt::space::host>(x, y, y_init);
  run_test_view_add<35, gt::space::host>(x, y, y_init);
  run_test_view_add<36, gt::space::host>(x, y, y_init);
  run_test_view_add<37, gt::space::host>(x, y, y_init);
  run_test_view_add<38, gt::space::host>(x, y, y_init);
  run_test_view_add<39, gt::space::host>(x, y, y_init);
  run_test_view_add<40, gt::space::host>(x, y, y_init);
  run_test_view_add<41, gt::space::host>(x, y, y_init);
  run_test_view_add<42, gt::space::host>(x, y, y_init);
  run_test_view_add<43, gt::space::host>(x, y, y_init);
  run_test_view_add<44, gt::space::host>(x, y, y_init);
  run_test_view_add<45, gt::space::host>(x, y, y_init);
  run_test_view_add<46, gt::space::host>(x, y, y_init);
  run_test_view_add<47, gt::space::host>(x, y, y_init);
  run_test_view_add<48, gt::space::host>(x, y, y_init);
  run_test_view_add<49, gt::space::host>(x, y, y_init);
  run_test_view_add<50, gt::space::host>(x, y, y_init);
  run_test_view_add<51, gt::space::host>(x, y, y_init);
  run_test_view_add<52, gt::space::host>(x, y, y_init);
  run_test_view_add<53, gt::space::host>(x, y, y_init);
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
  /* */ gt::gtensor<complex32_t, 1> y(ny, y_init);

  run_test_view_add<0, gt::space::host>(x, y, y_init);
  run_test_view_add<1, gt::space::host>(x, y, y_init);
  run_test_view_add<2, gt::space::host>(x, y, y_init);
  run_test_view_add<3, gt::space::host>(x, y, y_init);
  run_test_view_add<4, gt::space::host>(x, y, y_init);
  run_test_view_add<5, gt::space::host>(x, y, y_init);
  run_test_view_add<6, gt::space::host>(x, y, y_init);
  run_test_view_add<7, gt::space::host>(x, y, y_init);
  run_test_view_add<8, gt::space::host>(x, y, y_init);
  run_test_view_add<9, gt::space::host>(x, y, y_init);
  run_test_view_add<10, gt::space::host>(x, y, y_init);
  run_test_view_add<11, gt::space::host>(x, y, y_init);
  run_test_view_add<12, gt::space::host>(x, y, y_init);
  run_test_view_add<13, gt::space::host>(x, y, y_init);
  run_test_view_add<14, gt::space::host>(x, y, y_init);
  run_test_view_add<15, gt::space::host>(x, y, y_init);
  run_test_view_add<16, gt::space::host>(x, y, y_init);
  run_test_view_add<17, gt::space::host>(x, y, y_init);
  run_test_view_add<18, gt::space::host>(x, y, y_init);
  run_test_view_add<19, gt::space::host>(x, y, y_init);
  run_test_view_add<20, gt::space::host>(x, y, y_init);
  run_test_view_add<21, gt::space::host>(x, y, y_init);
  run_test_view_add<22, gt::space::host>(x, y, y_init);
  run_test_view_add<23, gt::space::host>(x, y, y_init);
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
  /* */ gt::gtensor<complex64_t, 1> y(ny, y_init);
  auto gt_y = gt::adapt<1>(y.data(), y.size());

  gt_y.view() = y_init;

  run_test_view_add<0, gt::space::host>(x, y, y_init);
  run_test_view_add<1, gt::space::host>(x, y, y_init);
  run_test_view_add<2, gt::space::host>(x, y, y_init);
  run_test_view_add<3, gt::space::host>(x, y, y_init);
  run_test_view_add<4, gt::space::host>(x, y, y_init);
  run_test_view_add<5, gt::space::host>(x, y, y_init);
  run_test_view_add<6, gt::space::host>(x, y, y_init);
  run_test_view_add<7, gt::space::host>(x, y, y_init);
  run_test_view_add<8, gt::space::host>(x, y, y_init);
  run_test_view_add<9, gt::space::host>(x, y, y_init);
  run_test_view_add<10, gt::space::host>(x, y, y_init);
  run_test_view_add<11, gt::space::host>(x, y, y_init);
  run_test_view_add<12, gt::space::host>(x, y, y_init);
  run_test_view_add<13, gt::space::host>(x, y, y_init);
  run_test_view_add<14, gt::space::host>(x, y, y_init);
  run_test_view_add<15, gt::space::host>(x, y, y_init);
  run_test_view_add<16, gt::space::host>(x, y, y_init);
  run_test_view_add<17, gt::space::host>(x, y, y_init);
  run_test_view_add<18, gt::space::host>(x, y, y_init);
  run_test_view_add<19, gt::space::host>(x, y, y_init);
  run_test_view_add<20, gt::space::host>(x, y, y_init);
  run_test_view_add<21, gt::space::host>(x, y, y_init);
  run_test_view_add<22, gt::space::host>(x, y, y_init);
  run_test_view_add<23, gt::space::host>(x, y, y_init);
  run_test_view_add<24, gt::space::host>(x, y, y_init);
  run_test_view_add<25, gt::space::host>(x, y, y_init);
  run_test_view_add<26, gt::space::host>(x, y, y_init);
  run_test_view_add<27, gt::space::host>(x, y, y_init);
  run_test_view_add<28, gt::space::host>(x, y, y_init);
  run_test_view_add<29, gt::space::host>(x, y, y_init);
  run_test_view_add<30, gt::space::host>(x, y, y_init);
  run_test_view_add<31, gt::space::host>(x, y, y_init);
  run_test_view_add<32, gt::space::host>(x, y, y_init);
  run_test_view_add<33, gt::space::host>(x, y, y_init);
  run_test_view_add<34, gt::space::host>(x, y, y_init);
  run_test_view_add<35, gt::space::host>(x, y, y_init);
  run_test_view_add<36, gt::space::host>(x, y, y_init);
  run_test_view_add<37, gt::space::host>(x, y, y_init);
  run_test_view_add<38, gt::space::host>(x, y, y_init);
  run_test_view_add<39, gt::space::host>(x, y, y_init);
  run_test_view_add<40, gt::space::host>(x, y, y_init);
  run_test_view_add<41, gt::space::host>(x, y, y_init);
  run_test_view_add<42, gt::space::host>(x, y, y_init);
  run_test_view_add<43, gt::space::host>(x, y, y_init);
  run_test_view_add<44, gt::space::host>(x, y, y_init);
  run_test_view_add<45, gt::space::host>(x, y, y_init);
  run_test_view_add<46, gt::space::host>(x, y, y_init);
  run_test_view_add<47, gt::space::host>(x, y, y_init);
  run_test_view_add<48, gt::space::host>(x, y, y_init);
  run_test_view_add<49, gt::space::host>(x, y, y_init);
  run_test_view_add<50, gt::space::host>(x, y, y_init);
  run_test_view_add<51, gt::space::host>(x, y, y_init);
  run_test_view_add<52, gt::space::host>(x, y, y_init);
  run_test_view_add<53, gt::space::host>(x, y, y_init);
}
