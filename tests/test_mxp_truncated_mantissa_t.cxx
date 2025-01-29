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

TEST(mxp_truncated_mantissa, add_float)
{
  const int n{3};

  const float x_init{1.f + exp2f(-13) + exp2f(-15) + exp2f(-16)};
  const float y_init{1.f};

  EXPECT_NE(y_init, y_init + x_init);

  const gt::gtensor<float, 1> x(n, x_init);
  /* */ gt::gtensor<float, 1> y(n, y_init);
  auto gt_y = gt::adapt<1>(y.data(), y.size());

  gt_y.view() = y_init;
  generic_truncated_add<0, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>(n, ref_truncated_add_float<0>())));

  gt_y.view() = y_init;
  generic_truncated_add<1, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>(n, ref_truncated_add_float<1>())));

  gt_y.view() = y_init;
  generic_truncated_add<2, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>(n, ref_truncated_add_float<2>())));

  gt_y.view() = y_init;
  generic_truncated_add<3, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>(n, ref_truncated_add_float<3>())));

  gt_y.view() = y_init;
  generic_truncated_add<4, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>(n, ref_truncated_add_float<4>())));

  gt_y.view() = y_init;
  generic_truncated_add<5, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>(n, ref_truncated_add_float<5>())));

  gt_y.view() = y_init;
  generic_truncated_add<6, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>(n, ref_truncated_add_float<6>())));

  gt_y.view() = y_init;
  generic_truncated_add<7, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>(n, ref_truncated_add_float<7>())));

  gt_y.view() = y_init;
  generic_truncated_add<8, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>(n, ref_truncated_add_float<8>())));

  gt_y.view() = y_init;
  generic_truncated_add<9, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>(n, ref_truncated_add_float<9>())));

  gt_y.view() = y_init;
  generic_truncated_add<10, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>(n, ref_truncated_add_float<10>())));

  gt_y.view() = y_init;
  generic_truncated_add<11, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>(n, ref_truncated_add_float<11>())));

  gt_y.view() = y_init;
  generic_truncated_add<12, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>(n, ref_truncated_add_float<12>())));

  gt_y.view() = y_init;
  generic_truncated_add<13, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>(n, ref_truncated_add_float<13>())));

  gt_y.view() = y_init;
  generic_truncated_add<14, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>(n, ref_truncated_add_float<14>())));

  gt_y.view() = y_init;
  generic_truncated_add<15, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>(n, ref_truncated_add_float<15>())));

  gt_y.view() = y_init;
  generic_truncated_add<16, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>(n, ref_truncated_add_float<16>())));

  gt_y.view() = y_init;
  generic_truncated_add<17, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>(n, ref_truncated_add_float<17>())));

  gt_y.view() = y_init;
  generic_truncated_add<18, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>(n, ref_truncated_add_float<18>())));

  gt_y.view() = y_init;
  generic_truncated_add<19, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>(n, ref_truncated_add_float<19>())));

  gt_y.view() = y_init;
  generic_truncated_add<20, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>(n, ref_truncated_add_float<20>())));

  gt_y.view() = y_init;
  generic_truncated_add<21, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>(n, ref_truncated_add_float<21>())));

  gt_y.view() = y_init;
  generic_truncated_add<22, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>(n, ref_truncated_add_float<22>())));

  gt_y.view() = y_init;
  generic_truncated_add<22, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>(n, ref_truncated_add_float<23>())));
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

TEST(mxp_truncated_mantissa, add_double)
{
  const int n{3};

  const double x_init{1. + exp2(-23) + exp2(-25) + exp2(-26)};
  const double y_init{1.};

  EXPECT_NE(y_init, y_init + x_init);

  const gt::gtensor<double, 1> x(n, x_init);
  /* */ gt::gtensor<double, 1> y(n, y_init);
  auto gt_y = gt::adapt<1>(y.data(), y.size());

  gt_y.view() = y_init;
  generic_truncated_add<0, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<0>())));

  gt_y.view() = y_init;
  generic_truncated_add<1, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<1>())));

  gt_y.view() = y_init;
  generic_truncated_add<2, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<2>())));

  gt_y.view() = y_init;
  generic_truncated_add<3, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<3>())));

  gt_y.view() = y_init;
  generic_truncated_add<4, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<4>())));

  gt_y.view() = y_init;
  generic_truncated_add<5, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<5>())));

  gt_y.view() = y_init;
  generic_truncated_add<6, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<6>())));

  gt_y.view() = y_init;
  generic_truncated_add<7, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<7>())));

  gt_y.view() = y_init;
  generic_truncated_add<8, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<8>())));

  gt_y.view() = y_init;
  generic_truncated_add<9, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<9>())));

  gt_y.view() = y_init;
  generic_truncated_add<10, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<10>())));

  gt_y.view() = y_init;
  generic_truncated_add<11, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<11>())));

  gt_y.view() = y_init;
  generic_truncated_add<12, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<12>())));

  gt_y.view() = y_init;
  generic_truncated_add<13, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<13>())));

  gt_y.view() = y_init;
  generic_truncated_add<14, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<14>())));

  gt_y.view() = y_init;
  generic_truncated_add<15, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<15>())));

  gt_y.view() = y_init;
  generic_truncated_add<16, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<16>())));

  gt_y.view() = y_init;
  generic_truncated_add<17, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<17>())));

  gt_y.view() = y_init;
  generic_truncated_add<18, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<18>())));

  gt_y.view() = y_init;
  generic_truncated_add<19, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<19>())));

  gt_y.view() = y_init;
  generic_truncated_add<20, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<20>())));

  gt_y.view() = y_init;
  generic_truncated_add<21, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<21>())));

  gt_y.view() = y_init;
  generic_truncated_add<22, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<22>())));

  gt_y.view() = y_init;
  generic_truncated_add<23, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<23>())));

  gt_y.view() = y_init;
  generic_truncated_add<24, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<24>())));

  gt_y.view() = y_init;
  generic_truncated_add<25, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<25>())));

  gt_y.view() = y_init;
  generic_truncated_add<26, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<26>())));

  gt_y.view() = y_init;
  generic_truncated_add<27, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<27>())));

  gt_y.view() = y_init;
  generic_truncated_add<28, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<28>())));

  gt_y.view() = y_init;
  generic_truncated_add<29, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<29>())));

  gt_y.view() = y_init;
  generic_truncated_add<30, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<30>())));

  gt_y.view() = y_init;
  generic_truncated_add<31, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<31>())));

  gt_y.view() = y_init;
  generic_truncated_add<32, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<32>())));

  gt_y.view() = y_init;
  generic_truncated_add<33, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<33>())));

  gt_y.view() = y_init;
  generic_truncated_add<34, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<34>())));

  gt_y.view() = y_init;
  generic_truncated_add<35, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<35>())));

  gt_y.view() = y_init;
  generic_truncated_add<36, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<36>())));

  gt_y.view() = y_init;
  generic_truncated_add<37, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<37>())));

  gt_y.view() = y_init;
  generic_truncated_add<38, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<38>())));

  gt_y.view() = y_init;
  generic_truncated_add<39, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<39>())));

  gt_y.view() = y_init;
  generic_truncated_add<40, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<40>())));

  gt_y.view() = y_init;
  generic_truncated_add<41, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<41>())));

  gt_y.view() = y_init;
  generic_truncated_add<42, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<42>())));

  gt_y.view() = y_init;
  generic_truncated_add<43, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<43>())));

  gt_y.view() = y_init;
  generic_truncated_add<44, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<44>())));

  gt_y.view() = y_init;
  generic_truncated_add<45, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<45>())));

  gt_y.view() = y_init;
  generic_truncated_add<46, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<46>())));

  gt_y.view() = y_init;
  generic_truncated_add<47, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<47>())));

  gt_y.view() = y_init;
  generic_truncated_add<48, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<48>())));

  gt_y.view() = y_init;
  generic_truncated_add<49, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<49>())));

  gt_y.view() = y_init;
  generic_truncated_add<50, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<50>())));

  gt_y.view() = y_init;
  generic_truncated_add<51, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<51>())));

  gt_y.view() = y_init;
  generic_truncated_add<52, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<52>())));

  gt_y.view() = y_init;
  generic_truncated_add<53, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>(n, ref_truncated_add_double<53>())));
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
  auto gt_y = gt::adapt<1>(y.data(), y.size());

  gt_y.view() = y_init;
  generic_truncated_add<0, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, ref_truncated_add_float<0>())));

  gt_y.view() = y_init;
  generic_truncated_add<1, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, ref_truncated_add_float<1>())));

  gt_y.view() = y_init;
  generic_truncated_add<2, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, ref_truncated_add_float<2>())));

  gt_y.view() = y_init;
  generic_truncated_add<3, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, ref_truncated_add_float<3>())));

  gt_y.view() = y_init;
  generic_truncated_add<4, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, ref_truncated_add_float<4>())));

  gt_y.view() = y_init;
  generic_truncated_add<5, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, ref_truncated_add_float<5>())));

  gt_y.view() = y_init;
  generic_truncated_add<6, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, ref_truncated_add_float<6>())));

  gt_y.view() = y_init;
  generic_truncated_add<7, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, ref_truncated_add_float<7>())));

  gt_y.view() = y_init;
  generic_truncated_add<8, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, ref_truncated_add_float<8>())));

  gt_y.view() = y_init;
  generic_truncated_add<9, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, ref_truncated_add_float<9>())));

  gt_y.view() = y_init;
  generic_truncated_add<10, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, ref_truncated_add_float<10>())));

  gt_y.view() = y_init;
  generic_truncated_add<11, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, ref_truncated_add_float<11>())));

  gt_y.view() = y_init;
  generic_truncated_add<12, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, ref_truncated_add_float<12>())));

  gt_y.view() = y_init;
  generic_truncated_add<13, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, ref_truncated_add_float<13>())));

  gt_y.view() = y_init;
  generic_truncated_add<14, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, ref_truncated_add_float<14>())));

  gt_y.view() = y_init;
  generic_truncated_add<15, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, ref_truncated_add_float<15>())));

  gt_y.view() = y_init;
  generic_truncated_add<16, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, ref_truncated_add_float<16>())));

  gt_y.view() = y_init;
  generic_truncated_add<17, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, ref_truncated_add_float<17>())));

  gt_y.view() = y_init;
  generic_truncated_add<18, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, ref_truncated_add_float<18>())));

  gt_y.view() = y_init;
  generic_truncated_add<19, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, ref_truncated_add_float<19>())));

  gt_y.view() = y_init;
  generic_truncated_add<20, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, ref_truncated_add_float<20>())));

  gt_y.view() = y_init;
  generic_truncated_add<21, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, ref_truncated_add_float<21>())));

  gt_y.view() = y_init;
  generic_truncated_add<22, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, ref_truncated_add_float<22>())));

  gt_y.view() = y_init;
  generic_truncated_add<23, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<complex32_t, 1>(n, ref_truncated_add_float<23>())));
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
  auto gt_y = gt::adapt<1>(y.data(), y.size());

  gt_y.view() = y_init;
  generic_truncated_add<0, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<0>())));

  gt_y.view() = y_init;
  generic_truncated_add<1, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<1>())));

  gt_y.view() = y_init;
  generic_truncated_add<2, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<2>())));

  gt_y.view() = y_init;
  generic_truncated_add<3, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<3>())));

  gt_y.view() = y_init;
  generic_truncated_add<4, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<4>())));

  gt_y.view() = y_init;
  generic_truncated_add<5, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<5>())));

  gt_y.view() = y_init;
  generic_truncated_add<6, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<6>())));

  gt_y.view() = y_init;
  generic_truncated_add<7, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<7>())));

  gt_y.view() = y_init;
  generic_truncated_add<8, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<8>())));

  gt_y.view() = y_init;
  generic_truncated_add<9, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<9>())));

  gt_y.view() = y_init;
  generic_truncated_add<10, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<10>())));

  gt_y.view() = y_init;
  generic_truncated_add<11, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<11>())));

  gt_y.view() = y_init;
  generic_truncated_add<12, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<12>())));

  gt_y.view() = y_init;
  generic_truncated_add<13, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<13>())));

  gt_y.view() = y_init;
  generic_truncated_add<14, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<14>())));

  gt_y.view() = y_init;
  generic_truncated_add<15, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<15>())));

  gt_y.view() = y_init;
  generic_truncated_add<16, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<16>())));

  gt_y.view() = y_init;
  generic_truncated_add<17, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<17>())));

  gt_y.view() = y_init;
  generic_truncated_add<18, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<18>())));

  gt_y.view() = y_init;
  generic_truncated_add<19, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<19>())));

  gt_y.view() = y_init;
  generic_truncated_add<20, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<20>())));

  gt_y.view() = y_init;
  generic_truncated_add<21, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<21>())));

  gt_y.view() = y_init;
  generic_truncated_add<22, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<22>())));

  gt_y.view() = y_init;
  generic_truncated_add<23, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<23>())));

  gt_y.view() = y_init;
  generic_truncated_add<24, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<24>())));

  gt_y.view() = y_init;
  generic_truncated_add<25, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<25>())));

  gt_y.view() = y_init;
  generic_truncated_add<26, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<26>())));

  gt_y.view() = y_init;
  generic_truncated_add<27, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<27>())));

  gt_y.view() = y_init;
  generic_truncated_add<28, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<28>())));

  gt_y.view() = y_init;
  generic_truncated_add<29, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<29>())));

  gt_y.view() = y_init;
  generic_truncated_add<30, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<30>())));

  gt_y.view() = y_init;
  generic_truncated_add<31, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<31>())));

  gt_y.view() = y_init;
  generic_truncated_add<32, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<32>())));

  gt_y.view() = y_init;
  generic_truncated_add<33, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<33>())));

  gt_y.view() = y_init;
  generic_truncated_add<34, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<34>())));

  gt_y.view() = y_init;
  generic_truncated_add<35, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<35>())));

  gt_y.view() = y_init;
  generic_truncated_add<36, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<36>())));

  gt_y.view() = y_init;
  generic_truncated_add<37, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<37>())));

  gt_y.view() = y_init;
  generic_truncated_add<38, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<38>())));

  gt_y.view() = y_init;
  generic_truncated_add<39, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<39>())));

  gt_y.view() = y_init;
  generic_truncated_add<40, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<40>())));

  gt_y.view() = y_init;
  generic_truncated_add<41, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<41>())));

  gt_y.view() = y_init;
  generic_truncated_add<42, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<42>())));

  gt_y.view() = y_init;
  generic_truncated_add<43, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<43>())));

  gt_y.view() = y_init;
  generic_truncated_add<44, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<44>())));

  gt_y.view() = y_init;
  generic_truncated_add<45, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<45>())));

  gt_y.view() = y_init;
  generic_truncated_add<46, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<46>())));

  gt_y.view() = y_init;
  generic_truncated_add<47, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<47>())));

  gt_y.view() = y_init;
  generic_truncated_add<48, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<48>())));

  gt_y.view() = y_init;
  generic_truncated_add<49, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<49>())));

  gt_y.view() = y_init;
  generic_truncated_add<50, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<50>())));

  gt_y.view() = y_init;
  generic_truncated_add<51, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<51>())));

  gt_y.view() = y_init;
  generic_truncated_add<52, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<52>())));

  gt_y.view() = y_init;
  generic_truncated_add<53, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<complex64_t, 1>(n, ref_truncated_add_double<53>())));
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

TEST(mxp_truncated_mantissa, view_add_float)
{
  const int nx{3};
  const int ny{5};

  const float x_init{1.f + exp2f(-13) + exp2f(-15) + exp2f(-16)};
  const float y_init{1.f};

  EXPECT_NE(y_init, y_init + x_init);

  const gt::gtensor<float, 1> x(nx, x_init);
  /* */ gt::gtensor<float, 1> y(ny, y_init);
  auto gt_y = gt::adapt<1>(y.data(), y.size());

  gt_y.view() = y_init;
  generic_view_truncated_add<0, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>{y_init, ref_truncated_add_float<0>(),
                                      ref_truncated_add_float<0>(),
                                      ref_truncated_add_float<0>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<1, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>{y_init, ref_truncated_add_float<1>(),
                                      ref_truncated_add_float<1>(),
                                      ref_truncated_add_float<1>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<2, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>{y_init, ref_truncated_add_float<2>(),
                                      ref_truncated_add_float<2>(),
                                      ref_truncated_add_float<2>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<3, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>{y_init, ref_truncated_add_float<3>(),
                                      ref_truncated_add_float<3>(),
                                      ref_truncated_add_float<3>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<4, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>{y_init, ref_truncated_add_float<4>(),
                                      ref_truncated_add_float<4>(),
                                      ref_truncated_add_float<4>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<5, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>{y_init, ref_truncated_add_float<5>(),
                                      ref_truncated_add_float<5>(),
                                      ref_truncated_add_float<5>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<6, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>{y_init, ref_truncated_add_float<6>(),
                                      ref_truncated_add_float<6>(),
                                      ref_truncated_add_float<6>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<7, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>{y_init, ref_truncated_add_float<7>(),
                                      ref_truncated_add_float<7>(),
                                      ref_truncated_add_float<7>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<8, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>{y_init, ref_truncated_add_float<8>(),
                                      ref_truncated_add_float<8>(),
                                      ref_truncated_add_float<8>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<9, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>{y_init, ref_truncated_add_float<9>(),
                                      ref_truncated_add_float<9>(),
                                      ref_truncated_add_float<9>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<10, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>{y_init, ref_truncated_add_float<10>(),
                                      ref_truncated_add_float<10>(),
                                      ref_truncated_add_float<10>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<11, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>{y_init, ref_truncated_add_float<11>(),
                                      ref_truncated_add_float<11>(),
                                      ref_truncated_add_float<11>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<12, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>{y_init, ref_truncated_add_float<12>(),
                                      ref_truncated_add_float<12>(),
                                      ref_truncated_add_float<12>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<13, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>{y_init, ref_truncated_add_float<13>(),
                                      ref_truncated_add_float<13>(),
                                      ref_truncated_add_float<13>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<14, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>{y_init, ref_truncated_add_float<14>(),
                                      ref_truncated_add_float<14>(),
                                      ref_truncated_add_float<14>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<15, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>{y_init, ref_truncated_add_float<15>(),
                                      ref_truncated_add_float<15>(),
                                      ref_truncated_add_float<15>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<16, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>{y_init, ref_truncated_add_float<16>(),
                                      ref_truncated_add_float<16>(),
                                      ref_truncated_add_float<16>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<17, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>{y_init, ref_truncated_add_float<17>(),
                                      ref_truncated_add_float<17>(),
                                      ref_truncated_add_float<17>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<18, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>{y_init, ref_truncated_add_float<18>(),
                                      ref_truncated_add_float<18>(),
                                      ref_truncated_add_float<18>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<19, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>{y_init, ref_truncated_add_float<19>(),
                                      ref_truncated_add_float<19>(),
                                      ref_truncated_add_float<19>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<20, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>{y_init, ref_truncated_add_float<20>(),
                                      ref_truncated_add_float<20>(),
                                      ref_truncated_add_float<20>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<21, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>{y_init, ref_truncated_add_float<21>(),
                                      ref_truncated_add_float<21>(),
                                      ref_truncated_add_float<21>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<22, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>{y_init, ref_truncated_add_float<22>(),
                                      ref_truncated_add_float<22>(),
                                      ref_truncated_add_float<22>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<23, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<float, 1>{y_init, ref_truncated_add_float<23>(),
                                      ref_truncated_add_float<23>(),
                                      ref_truncated_add_float<23>(), y_init}));
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
  auto gt_y = gt::adapt<1>(y.data(), y.size());

  gt_y.view() = y_init;
  generic_view_truncated_add<0, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<0>(),
                                       ref_truncated_add_double<0>(),
                                       ref_truncated_add_double<0>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<1, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<1>(),
                                       ref_truncated_add_double<1>(),
                                       ref_truncated_add_double<1>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<2, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<2>(),
                                       ref_truncated_add_double<2>(),
                                       ref_truncated_add_double<2>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<3, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<3>(),
                                       ref_truncated_add_double<3>(),
                                       ref_truncated_add_double<3>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<4, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<4>(),
                                       ref_truncated_add_double<4>(),
                                       ref_truncated_add_double<4>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<5, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<5>(),
                                       ref_truncated_add_double<5>(),
                                       ref_truncated_add_double<5>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<6, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<6>(),
                                       ref_truncated_add_double<6>(),
                                       ref_truncated_add_double<6>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<7, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<7>(),
                                       ref_truncated_add_double<7>(),
                                       ref_truncated_add_double<7>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<8, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<8>(),
                                       ref_truncated_add_double<8>(),
                                       ref_truncated_add_double<8>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<9, gt::space::host>(x, y);
  EXPECT_EQ(y, (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<9>(),
                                       ref_truncated_add_double<9>(),
                                       ref_truncated_add_double<9>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<10, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<10>(),
                                    ref_truncated_add_double<10>(),
                                    ref_truncated_add_double<10>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<11, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<11>(),
                                    ref_truncated_add_double<11>(),
                                    ref_truncated_add_double<11>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<12, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<12>(),
                                    ref_truncated_add_double<12>(),
                                    ref_truncated_add_double<12>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<13, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<13>(),
                                    ref_truncated_add_double<13>(),
                                    ref_truncated_add_double<13>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<14, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<14>(),
                                    ref_truncated_add_double<14>(),
                                    ref_truncated_add_double<14>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<15, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<15>(),
                                    ref_truncated_add_double<15>(),
                                    ref_truncated_add_double<15>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<16, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<16>(),
                                    ref_truncated_add_double<16>(),
                                    ref_truncated_add_double<16>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<17, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<17>(),
                                    ref_truncated_add_double<17>(),
                                    ref_truncated_add_double<17>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<18, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<18>(),
                                    ref_truncated_add_double<18>(),
                                    ref_truncated_add_double<18>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<19, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<19>(),
                                    ref_truncated_add_double<19>(),
                                    ref_truncated_add_double<19>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<20, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<20>(),
                                    ref_truncated_add_double<20>(),
                                    ref_truncated_add_double<20>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<21, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<21>(),
                                    ref_truncated_add_double<21>(),
                                    ref_truncated_add_double<21>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<22, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<22>(),
                                    ref_truncated_add_double<22>(),
                                    ref_truncated_add_double<22>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<23, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<23>(),
                                    ref_truncated_add_double<23>(),
                                    ref_truncated_add_double<23>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<24, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<24>(),
                                    ref_truncated_add_double<24>(),
                                    ref_truncated_add_double<24>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<25, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<25>(),
                                    ref_truncated_add_double<25>(),
                                    ref_truncated_add_double<25>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<26, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<26>(),
                                    ref_truncated_add_double<26>(),
                                    ref_truncated_add_double<26>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<27, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<27>(),
                                    ref_truncated_add_double<27>(),
                                    ref_truncated_add_double<27>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<28, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<28>(),
                                    ref_truncated_add_double<28>(),
                                    ref_truncated_add_double<28>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<29, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<29>(),
                                    ref_truncated_add_double<29>(),
                                    ref_truncated_add_double<29>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<30, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<30>(),
                                    ref_truncated_add_double<30>(),
                                    ref_truncated_add_double<30>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<31, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<31>(),
                                    ref_truncated_add_double<31>(),
                                    ref_truncated_add_double<31>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<32, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<32>(),
                                    ref_truncated_add_double<32>(),
                                    ref_truncated_add_double<32>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<33, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<33>(),
                                    ref_truncated_add_double<33>(),
                                    ref_truncated_add_double<33>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<34, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<34>(),
                                    ref_truncated_add_double<34>(),
                                    ref_truncated_add_double<34>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<35, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<35>(),
                                    ref_truncated_add_double<35>(),
                                    ref_truncated_add_double<35>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<36, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<36>(),
                                    ref_truncated_add_double<36>(),
                                    ref_truncated_add_double<36>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<37, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<37>(),
                                    ref_truncated_add_double<37>(),
                                    ref_truncated_add_double<37>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<38, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<38>(),
                                    ref_truncated_add_double<38>(),
                                    ref_truncated_add_double<38>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<39, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<39>(),
                                    ref_truncated_add_double<39>(),
                                    ref_truncated_add_double<39>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<40, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<40>(),
                                    ref_truncated_add_double<40>(),
                                    ref_truncated_add_double<40>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<41, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<41>(),
                                    ref_truncated_add_double<41>(),
                                    ref_truncated_add_double<41>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<42, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<42>(),
                                    ref_truncated_add_double<42>(),
                                    ref_truncated_add_double<42>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<43, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<43>(),
                                    ref_truncated_add_double<43>(),
                                    ref_truncated_add_double<43>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<44, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<44>(),
                                    ref_truncated_add_double<44>(),
                                    ref_truncated_add_double<44>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<45, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<45>(),
                                    ref_truncated_add_double<45>(),
                                    ref_truncated_add_double<45>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<46, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<46>(),
                                    ref_truncated_add_double<46>(),
                                    ref_truncated_add_double<46>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<47, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<47>(),
                                    ref_truncated_add_double<47>(),
                                    ref_truncated_add_double<47>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<48, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<48>(),
                                    ref_truncated_add_double<48>(),
                                    ref_truncated_add_double<48>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<49, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<49>(),
                                    ref_truncated_add_double<49>(),
                                    ref_truncated_add_double<49>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<50, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<50>(),
                                    ref_truncated_add_double<50>(),
                                    ref_truncated_add_double<50>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<51, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<51>(),
                                    ref_truncated_add_double<51>(),
                                    ref_truncated_add_double<51>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<52, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<52>(),
                                    ref_truncated_add_double<52>(),
                                    ref_truncated_add_double<52>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<53, gt::space::host>(x, y);
  EXPECT_EQ(y,
            (gt::gtensor<double, 1>{y_init, ref_truncated_add_double<53>(),
                                    ref_truncated_add_double<53>(),
                                    ref_truncated_add_double<53>(), y_init}));
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
  auto gt_y = gt::adapt<1>(y.data(), y.size());

  gt_y.view() = y_init;
  generic_view_truncated_add<0, gt::space::host>(x, y);
  EXPECT_EQ(
    y, (gt::gtensor<complex32_t, 1>{y_init, ref_truncated_add_float<0>(),
                                    ref_truncated_add_float<0>(),
                                    ref_truncated_add_float<0>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<1, gt::space::host>(x, y);
  EXPECT_EQ(
    y, (gt::gtensor<complex32_t, 1>{y_init, ref_truncated_add_float<1>(),
                                    ref_truncated_add_float<1>(),
                                    ref_truncated_add_float<1>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<2, gt::space::host>(x, y);
  EXPECT_EQ(
    y, (gt::gtensor<complex32_t, 1>{y_init, ref_truncated_add_float<2>(),
                                    ref_truncated_add_float<2>(),
                                    ref_truncated_add_float<2>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<3, gt::space::host>(x, y);
  EXPECT_EQ(
    y, (gt::gtensor<complex32_t, 1>{y_init, ref_truncated_add_float<3>(),
                                    ref_truncated_add_float<3>(),
                                    ref_truncated_add_float<3>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<4, gt::space::host>(x, y);
  EXPECT_EQ(
    y, (gt::gtensor<complex32_t, 1>{y_init, ref_truncated_add_float<4>(),
                                    ref_truncated_add_float<4>(),
                                    ref_truncated_add_float<4>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<5, gt::space::host>(x, y);
  EXPECT_EQ(
    y, (gt::gtensor<complex32_t, 1>{y_init, ref_truncated_add_float<5>(),
                                    ref_truncated_add_float<5>(),
                                    ref_truncated_add_float<5>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<6, gt::space::host>(x, y);
  EXPECT_EQ(
    y, (gt::gtensor<complex32_t, 1>{y_init, ref_truncated_add_float<6>(),
                                    ref_truncated_add_float<6>(),
                                    ref_truncated_add_float<6>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<7, gt::space::host>(x, y);
  EXPECT_EQ(
    y, (gt::gtensor<complex32_t, 1>{y_init, ref_truncated_add_float<7>(),
                                    ref_truncated_add_float<7>(),
                                    ref_truncated_add_float<7>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<8, gt::space::host>(x, y);
  EXPECT_EQ(
    y, (gt::gtensor<complex32_t, 1>{y_init, ref_truncated_add_float<8>(),
                                    ref_truncated_add_float<8>(),
                                    ref_truncated_add_float<8>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<9, gt::space::host>(x, y);
  EXPECT_EQ(
    y, (gt::gtensor<complex32_t, 1>{y_init, ref_truncated_add_float<9>(),
                                    ref_truncated_add_float<9>(),
                                    ref_truncated_add_float<9>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<10, gt::space::host>(x, y);
  EXPECT_EQ(
    y, (gt::gtensor<complex32_t, 1>{y_init, ref_truncated_add_float<10>(),
                                    ref_truncated_add_float<10>(),
                                    ref_truncated_add_float<10>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<11, gt::space::host>(x, y);
  EXPECT_EQ(
    y, (gt::gtensor<complex32_t, 1>{y_init, ref_truncated_add_float<11>(),
                                    ref_truncated_add_float<11>(),
                                    ref_truncated_add_float<11>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<12, gt::space::host>(x, y);
  EXPECT_EQ(
    y, (gt::gtensor<complex32_t, 1>{y_init, ref_truncated_add_float<12>(),
                                    ref_truncated_add_float<12>(),
                                    ref_truncated_add_float<12>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<13, gt::space::host>(x, y);
  EXPECT_EQ(
    y, (gt::gtensor<complex32_t, 1>{y_init, ref_truncated_add_float<13>(),
                                    ref_truncated_add_float<13>(),
                                    ref_truncated_add_float<13>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<14, gt::space::host>(x, y);
  EXPECT_EQ(
    y, (gt::gtensor<complex32_t, 1>{y_init, ref_truncated_add_float<14>(),
                                    ref_truncated_add_float<14>(),
                                    ref_truncated_add_float<14>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<15, gt::space::host>(x, y);
  EXPECT_EQ(
    y, (gt::gtensor<complex32_t, 1>{y_init, ref_truncated_add_float<15>(),
                                    ref_truncated_add_float<15>(),
                                    ref_truncated_add_float<15>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<16, gt::space::host>(x, y);
  EXPECT_EQ(
    y, (gt::gtensor<complex32_t, 1>{y_init, ref_truncated_add_float<16>(),
                                    ref_truncated_add_float<16>(),
                                    ref_truncated_add_float<16>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<17, gt::space::host>(x, y);
  EXPECT_EQ(
    y, (gt::gtensor<complex32_t, 1>{y_init, ref_truncated_add_float<17>(),
                                    ref_truncated_add_float<17>(),
                                    ref_truncated_add_float<17>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<18, gt::space::host>(x, y);
  EXPECT_EQ(
    y, (gt::gtensor<complex32_t, 1>{y_init, ref_truncated_add_float<18>(),
                                    ref_truncated_add_float<18>(),
                                    ref_truncated_add_float<18>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<19, gt::space::host>(x, y);
  EXPECT_EQ(
    y, (gt::gtensor<complex32_t, 1>{y_init, ref_truncated_add_float<19>(),
                                    ref_truncated_add_float<19>(),
                                    ref_truncated_add_float<19>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<20, gt::space::host>(x, y);
  EXPECT_EQ(
    y, (gt::gtensor<complex32_t, 1>{y_init, ref_truncated_add_float<20>(),
                                    ref_truncated_add_float<20>(),
                                    ref_truncated_add_float<20>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<21, gt::space::host>(x, y);
  EXPECT_EQ(
    y, (gt::gtensor<complex32_t, 1>{y_init, ref_truncated_add_float<21>(),
                                    ref_truncated_add_float<21>(),
                                    ref_truncated_add_float<21>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<22, gt::space::host>(x, y);
  EXPECT_EQ(
    y, (gt::gtensor<complex32_t, 1>{y_init, ref_truncated_add_float<22>(),
                                    ref_truncated_add_float<22>(),
                                    ref_truncated_add_float<22>(), y_init}));

  gt_y.view() = y_init;
  generic_view_truncated_add<23, gt::space::host>(x, y);
  EXPECT_EQ(
    y, (gt::gtensor<complex32_t, 1>{y_init, ref_truncated_add_float<23>(),
                                    ref_truncated_add_float<23>(),
                                    ref_truncated_add_float<23>(), y_init}));
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
  generic_view_truncated_add<0, gt::space::host>(x, y);
  EXPECT_EQ(
    y, (gt::gtensor<complex64_t, 1>{y_init, ref_truncated_add_double<0>(),
                                    ref_truncated_add_double<0>(),
                                    ref_truncated_add_double<0>(), y_init}));
}
