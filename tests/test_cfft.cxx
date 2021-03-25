#include <gtest/gtest.h>

#include "gtensor/gtensor.h"

#include "gt-fft/cfft.h"

#include "test_helpers.h"

template <typename E, typename New, typename Exec, typename Del>
void fft_r2c_1d(New&& newf, Exec&& execf, Del&& delf)
{
  constexpr int N = 4;
  constexpr int Nout = N / 2 + 1;
  constexpr int RANK = 1;
  constexpr int batch_size = 2;
  using T = gt::complex<E>;
  int lengths[RANK] = {N};

  gt::gtensor<E, 2> h_A(gt::shape(N, batch_size));
  gt::gtensor_device<E, 2> d_A(gt::shape(N, batch_size));

  gt::gtensor<T, 2> h_B(gt::shape(Nout, batch_size));
  gt::gtensor_device<T, 2> d_B(gt::shape(Nout, batch_size));

  // x = [2 3 -1 4];
  h_A(0, 0) = 2;
  h_A(1, 0) = 3;
  h_A(2, 0) = -1;
  h_A(3, 0) = 4;

  // y = [7 -21 11 1];
  h_A(0, 1) = 7;
  h_A(1, 1) = -21;
  h_A(2, 1) = 11;
  h_A(3, 1) = 1;

  // zero output array, rocfft at least does not zero padding elements
  // for real to complex transform
  gt::backend::device_memset(gt::backend::raw_pointer_cast(d_B.data()), 0,
                             batch_size * Nout * sizeof(T*));

  gt::copy(h_A, d_A);

  auto p = newf(1, lengths, 1, N, 1, Nout, batch_size);

  // fft(x) -> [8+0i 3+1i -6+0i 3-1i]
  // but with fftw convention for real transforms, the last term is
  // conjugate of second and set to 0 to save storage / computation
  execf(p, gt::backend::raw_pointer_cast(d_A.data()),
        gt::backend::raw_pointer_cast(d_B.data()));

  delf(p);

  gt::copy(d_B, h_B);

  expect_complex_near(h_B(0, 0), T(8, 0));
  expect_complex_near(h_B(1, 0), T(3, 1));
  expect_complex_near(h_B(2, 0), T(-6, 0));

  expect_complex_near(h_B(0, 1), T(-2, 0));
  expect_complex_near(h_B(1, 1), T(-4, 22));
  expect_complex_near(h_B(2, 1), T(38, 0));
}

TEST(cfft, d2z_1d)
{
  fft_r2c_1d<double>(&gtfft_new_real_double, &gtfft_dz,
                     &gtfft_delete_real_double);
}

TEST(cfft, r2c_1d)
{
  fft_r2c_1d<float>(&gtfft_new_real_float, &gtfft_rc, &gtfft_delete_real_float);
}

template <typename E, typename New, typename Exec, typename Del>
void fft_c2r_1d(New&& newf, Exec&& execf, Del&& delf)
{
  constexpr int N = 4;
  constexpr int Ncomplex = N / 2 + 1;
  constexpr int RANK = 1;
  constexpr int batch_size = 2;
  using T = gt::complex<E>;
  int lengths[RANK] = {N};

  gt::gtensor<E, 2> h_A(gt::shape(N, batch_size));
  gt::gtensor_device<E, 2> d_A(gt::shape(N, batch_size));

  gt::gtensor<T, 2> h_B(gt::shape(Ncomplex, batch_size));
  gt::gtensor_device<T, 2> d_B(gt::shape(Ncomplex, batch_size));

  h_B(0, 0) = T(8, 0);
  h_B(1, 0) = T(3, 1);
  h_B(2, 0) = T(-6, 0);

  h_B(0, 1) = T(-2, 0);
  h_B(1, 1) = T(-4, 22);
  h_B(2, 1) = T(38, 0);

  gt::copy(h_B, d_B);

  // ifft(x) -> [8+0i 3+1i -6+0i 3-1i]
  auto p = newf(1, lengths, 1, N, 1, Ncomplex, batch_size);

  execf(p, gt::backend::raw_pointer_cast(d_B.data()),
        gt::backend::raw_pointer_cast(d_A.data()));

  delf(p);

  gt::copy(d_A, h_A);

  EXPECT_EQ(h_A(0, 0) / N, 2);
  EXPECT_EQ(h_A(1, 0) / N, 3);
  EXPECT_EQ(h_A(2, 0) / N, -1);
  EXPECT_EQ(h_A(3, 0) / N, 4);

  EXPECT_EQ(h_A(0, 1) / N, 7);
  EXPECT_EQ(h_A(1, 1) / N, -21);
  EXPECT_EQ(h_A(2, 1) / N, 11);
  EXPECT_EQ(h_A(3, 1) / N, 1);
}

TEST(cfft, z2d_1d)
{
  fft_c2r_1d<double>(&gtfft_new_real_double, &gtfft_inverse_zd,
                     &gtfft_delete_real_double);
}

TEST(cfft, c2r_1d)
{
  fft_c2r_1d<float>(&gtfft_new_real_float, &gtfft_inverse_cr,
                    &gtfft_delete_real_float);
}

template <typename E, typename New, typename Exec, typename Del>
void fft_c2c_1d_forward(New&& newf, Exec&& execf, Del&& delf)
{
  constexpr int N = 4;
  constexpr int RANK = 1;
  constexpr int batch_size = 2;
  using T = gt::complex<E>;
  int lengths[RANK] = {N};

  gt::gtensor<T, 2> h_A(gt::shape(N, batch_size));
  gt::gtensor_device<T, 2> d_A(gt::shape(N, batch_size));

  gt::gtensor<T, 2> h_B(gt::shape(N, batch_size));
  gt::gtensor_device<T, 2> d_B(gt::shape(N, batch_size));

  // x = [2 3 -1 4];
  h_A(0, 0) = 2;
  h_A(1, 0) = 3;
  h_A(2, 0) = -1;
  h_A(3, 0) = 4;

  // y = [7 -21 11 1];
  h_A(0, 1) = 7;
  h_A(1, 1) = -21;
  h_A(2, 1) = 11;
  h_A(3, 1) = 1;

  gt::copy(h_A, d_A);

  auto p = newf(1, lengths, 1, N, 1, N, batch_size);

  execf(p, gt::backend::raw_pointer_cast(d_A.data()),
        gt::backend::raw_pointer_cast(d_B.data()));

  delf(p);

  gt::copy(d_B, h_B);

  expect_complex_near(h_B(0, 0), T(8, 0));
  expect_complex_near(h_B(1, 0), T(3, 1));
  expect_complex_near(h_B(2, 0), T(-6, 0));
  expect_complex_near(h_B(3, 0), T(3, -1));

  expect_complex_near(h_B(0, 1), T(-2, 0));
  expect_complex_near(h_B(1, 1), T(-4, 22));
  expect_complex_near(h_B(2, 1), T(38, 0));
  expect_complex_near(h_B(3, 1), T(-4, -22));
}

TEST(cfft, z2z_1d_forward)
{
  fft_c2c_1d_forward<double>(&gtfft_new_complex_double, &gtfft_zz,
                             &gtfft_delete_complex_double);
}

TEST(cfft, c2c_1d_forward)
{
  fft_c2c_1d_forward<float>(&gtfft_new_complex_float, &gtfft_cc,
                            &gtfft_delete_complex_float);
}

template <typename E, typename New, typename Exec, typename Del>
void fft_c2c_1d_inverse(New&& newf, Exec&& execf, Del&& delf)
{
  constexpr int N = 4;
  constexpr int RANK = 1;
  constexpr int batch_size = 2;
  using T = gt::complex<E>;
  int lengths[RANK] = {N};

  gt::gtensor<T, 2> h_A(gt::shape(N, batch_size));
  gt::gtensor_device<T, 2> d_A(gt::shape(N, batch_size));

  gt::gtensor<T, 2> h_B(gt::shape(N, batch_size));
  gt::gtensor_device<T, 2> d_B(gt::shape(N, batch_size));

  h_A(0, 0) = T(8, 0);
  h_A(1, 0) = T(3, 1);
  h_A(2, 0) = T(-6, 0);
  h_A(3, 0) = T(3, -1);

  h_A(0, 1) = T(-2, 0);
  h_A(1, 1) = T(-4, 22);
  h_A(2, 1) = T(38, 0);
  h_A(3, 1) = T(-4, -22);

  gt::copy(h_A, d_A);

  // ifft(x) -> [8+0i 3+1i -6+0i 3-1i]
  auto p = newf(1, lengths, 1, N, 1, N, batch_size);

  execf(p, gt::backend::raw_pointer_cast(d_A.data()),
        gt::backend::raw_pointer_cast(d_B.data()));

  delf(p);

  gt::copy(d_B, h_B);

  // required when using std::complex, int multiply is not defined
  auto dN = static_cast<E>(N);
  expect_complex_near(h_B(0, 0), dN * T(2, 0));
  expect_complex_near(h_B(1, 0), dN * T(3, 0));
  expect_complex_near(h_B(2, 0), dN * T(-1, 0));
  expect_complex_near(h_B(3, 0), dN * T(4, 0));

  expect_complex_near(h_B(0, 1), dN * T(7, 0));
  expect_complex_near(h_B(1, 1), dN * T(-21, 0));
  expect_complex_near(h_B(2, 1), dN * T(11, 0));
  expect_complex_near(h_B(3, 1), dN * T(1, 0));
}

TEST(cfft, z2z_1d_inverse)
{
  fft_c2c_1d_inverse<double>(&gtfft_new_complex_double, &gtfft_inverse_zz,
                             &gtfft_delete_complex_double);
}

TEST(cfft, c2c_1d_inverse)
{
  fft_c2c_1d_inverse<float>(&gtfft_new_complex_float, &gtfft_inverse_cc,
                            &gtfft_delete_complex_float);
}
