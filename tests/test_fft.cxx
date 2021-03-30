#include <gtest/gtest.h>

#include <stdexcept>

#include "gtensor/gtensor.h"

#include "gt-fft/fft.h"

#include "test_helpers.h"

template <typename E>
void fft_r2c_1d()
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

  // fft(x) -> [8+0i 3+1i -6+0i 3-1i]
  // but with fftw convention for real transforms, the last term is
  // conjugate of second and set to 0 to save storage / computation
  gt::fft::FFTPlanMany<gt::fft::Domain::REAL, E> plan(1, lengths, 1, N, 1, Nout,
                                                      batch_size);
  // plan.exec_forward(gt::backend::raw_pointer_cast(d_A.data()),
  //                  gt::backend::raw_pointer_cast(d_B.data()));
  plan(d_A, d_B);

  gt::copy(d_B, h_B);

  expect_complex_near(h_B(0, 0), T(8, 0));
  expect_complex_near(h_B(1, 0), T(3, 1));
  expect_complex_near(h_B(2, 0), T(-6, 0));

  expect_complex_near(h_B(0, 1), T(-2, 0));
  expect_complex_near(h_B(1, 1), T(-4, 22));
  expect_complex_near(h_B(2, 1), T(38, 0));
}

TEST(fft, d2z_1d)
{
  fft_r2c_1d<double>();
}

TEST(fft, r2c_1d)
{
  fft_r2c_1d<float>();
}

template <typename E>
void fft_c2r_1d()
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
  gt::fft::FFTPlanMany<gt::fft::Domain::REAL, E> plan(1, lengths, 1, N, 1,
                                                      Ncomplex, batch_size);
  // plan.exec_inverse(gt::backend::raw_pointer_cast(d_B.data()),
  //                  gt::backend::raw_pointer_cast(d_A.data()));
  plan.inverse(d_B, d_A);

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

TEST(fft, z2d_1d)
{
  fft_c2r_1d<double>();
}

TEST(fft, c2r_1d)
{
  fft_c2r_1d<float>();
}
template <typename E>
void fft_c2c_1d_forward()
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

  gt::fft::FFTPlanMany<gt::fft::Domain::COMPLEX, E> plan(1, lengths, 1, N, 1, N,
                                                         batch_size);

  // ifft(x) -> [8+0i 3+1i -6+0i 3-1i]
  // plan.exec_forward(gt::backend::raw_pointer_cast(d_A.data()),
  //                  gt::backend::raw_pointer_cast(d_B.data()));
  plan(d_A, d_B);

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

TEST(fft, z2z_1d_forward)
{
  fft_c2c_1d_forward<double>();
}

TEST(fft, c2c_1d_forward)
{
  fft_c2c_1d_forward<float>();
}

template <typename E>
void fft_c2c_1d_inverse()
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

  gt::fft::FFTPlanMany<gt::fft::Domain::COMPLEX, E> plan(1, lengths, 1, N, 1, N,
                                                         batch_size);

  // ifft(x) -> [8+0i 3+1i -6+0i 3-1i]
  // plan.exec_inverse(gt::backend::raw_pointer_cast(d_A.data()),
  //                  gt::backend::raw_pointer_cast(d_B.data()));
  plan.inverse(d_A, d_B);

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

TEST(fft, z2z_1d_inverse)
{
  fft_c2c_1d_inverse<double>();
}

TEST(fft, c2c_1d_inverse)
{
  fft_c2c_1d_inverse<float>();
}

TEST(fft, move_only)
{
  constexpr int N = 4;
  constexpr int Nout = N / 2 + 1;
  constexpr int RANK = 1;
  constexpr int batch_size = 1;
  using E = double;
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

  // zero output array, rocfft at least does not zero padding elements
  // for real to complex transform
  gt::backend::device_memset(gt::backend::raw_pointer_cast(d_B.data()), 0,
                             batch_size * Nout * sizeof(T*));

  gt::copy(h_A, d_A);

  // fft(x) -> [8+0i 3+1i -6+0i 3-1i]
  // but with fftw convention for real transforms, the last term is
  // conjugate of second and set to 0 to save storage / computation
  gt::fft::FFTPlanMany<gt::fft::Domain::REAL, E> plan(1, lengths, 1, N, 1, Nout,
                                                      batch_size);

  // Should not compile
  //  auto plan_copy = plan;
  //  gt::fft::FFTPlanMany<gt::fft::Domain::REAL, E> plan_copy2(plan);

  // do a move, then try to execute
  auto plan_moved = std::move(plan);

  // original plan is not valid, should throw error
  EXPECT_THROW(plan(d_A, d_B), std::runtime_error);

  plan_moved(d_A, d_B);

  gt::copy(d_B, h_B);

  expect_complex_near(h_B(0, 0), T(8, 0));
  expect_complex_near(h_B(1, 0), T(3, 1));
  expect_complex_near(h_B(2, 0), T(-6, 0));
}
