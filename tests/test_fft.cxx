#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

#include "gtensor/gtensor.h"

#include "gt-fft/fft.h"

#include "test_helpers.h"

constexpr double PI = 3.141592653589793;

template <typename E>
void fft_r2c_1d()
{
  constexpr int N = 4;
  constexpr int Nout = N / 2 + 1;
  constexpr int batch_size = 2;
  using T = gt::complex<E>;

  gt::gtensor<E, 2> h_A(gt::shape(N, batch_size));
  gt::gtensor_device<E, 2> d_A(gt::shape(N, batch_size));

  gt::gtensor<E, 2> h_A2(gt::shape(N, batch_size));
  gt::gtensor_device<E, 2> d_A2(gt::shape(N, batch_size));

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
  gt::backend::fill<gt::space::device>(
    gt::backend::raw_pointer_cast(d_B.data()),
    gt::backend::raw_pointer_cast(d_B.data()) + batch_size * Nout, 0);

  gt::copy(h_A, d_A);

  // fft(x) -> [8+0i 3+1i -6+0i 3-1i]
  // but with fftw convention for real transforms, the last term is
  // conjugate of second and set to 0 to save storage / computation
  gt::fft::FFTPlanMany<gt::fft::Domain::REAL, E> plan({N}, batch_size);
  // plan.exec_forward(gt::backend::raw_pointer_cast(d_A.data()),
  //                  gt::backend::raw_pointer_cast(d_B.data()));
  plan(d_A, d_B);

  // test roundtripping data
  plan.inverse(d_B, d_A2);

  gt::copy(d_B, h_B);
  gt::copy(d_A2, h_A2);

  EXPECT_EQ(h_A, h_A2 / N);

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
  constexpr int batch_size = 2;
  using T = gt::complex<E>;

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
  gt::fft::FFTPlanMany<gt::fft::Domain::REAL, E> plan({N}, batch_size);
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
  constexpr int batch_size = 2;
  using T = gt::complex<E>;

  gt::gtensor<T, 2> h_A(gt::shape(N, batch_size));
  gt::gtensor_device<T, 2> d_A(gt::shape(N, batch_size));

  gt::gtensor<T, 2> h_A2(h_A.shape());
  gt::gtensor_device<T, 2> d_A2(h_A.shape());

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

  gt::fft::FFTPlanMany<gt::fft::Domain::COMPLEX, E> plan({N}, batch_size);

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

  // test round trip
  plan.inverse(d_B, d_A2);
  gt::copy(d_A2, h_A2);

  for (int i = 0; i < h_A.shape(1); i++) {
    for (int j = 0; j < h_A.shape(0); j++) {
      expect_complex_near(h_A(j, i), h_A2(j, i) / T(N, 0));
    }
  }
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
  constexpr int batch_size = 2;
  using T = gt::complex<E>;

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

  gt::fft::FFTPlanMany<gt::fft::Domain::COMPLEX, E> plan({N}, batch_size);

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
  constexpr int batch_size = 1;
  using E = double;
  using T = gt::complex<E>;

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
  gt::backend::fill<gt::space::device>(
    gt::backend::raw_pointer_cast(d_B.data()),
    gt::backend::raw_pointer_cast(d_B.data()) + batch_size * Nout, 0);

  gt::copy(h_A, d_A);

  // fft(x) -> [8+0i 3+1i -6+0i 3-1i]
  // but with fftw convention for real transforms, the last term is
  // conjugate of second and set to 0 to save storage / computation
  gt::fft::FFTPlanMany<gt::fft::Domain::REAL, E> plan({N}, batch_size);

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

template <typename E>
void fft_r2c_2d()
{
  // sin(2*pi*x + 4*pi*y) x=-2:1/16:2-1/16, y=0:1/16:1-1/16
  constexpr int Nx = 64;
  constexpr int Ny = 16;
  constexpr int batch_size = 1;
  using T = gt::complex<E>;

  auto h_A = gt::zeros<E>({Nx, Ny, batch_size});
  auto d_A = gt::empty_device<E>(h_A.shape());

  auto h_A2 = gt::zeros<E>(h_A.shape());
  auto d_A2 = gt::empty_device<E>(h_A.shape());

  auto h_B = gt::empty<T>({Nx / 2 + 1, Ny, batch_size});
  auto d_B = gt::empty_device<T>(h_B.shape());

  // origin at center of domain has value 1, to model delta function
  double x, y;
  for (int j = 0; j < Ny; j++) {
    for (int i = 0; i < Nx; i++) {
      x = -2.0 + i / 16.0;
      y = j / 16.0;
      h_A(i, j, 0) = sin(2 * PI * x + 4 * PI * y);
    }
  }

  gt::copy(h_A, d_A);

  gt::fft::FFTPlanMany<gt::fft::Domain::REAL, E> plan({Ny, Nx}, batch_size);
  plan(d_A, d_B);
  gt::copy(d_B, h_B);

  /*
  std::cout << "h_A.shape() = " << h_A.shape() << std::endl;
  std::cout << "h_B.shape() = " << h_B.shape() << std::endl;
  std::cout << "h_A = " << h_A << std::endl;
  std::cout << "h_B = " << h_B << std::endl;
  */

  // sin over our domain has frequency 4 in x and 2 in y
  // NB: allow greater error than for other tests
  double max_err = 20.0 * gt::test::detail::max_err<E>::value;
  for (int j = 0; j < h_B.shape(1); j++) {
    for (int i = 0; i < h_B.shape(0); i++) {
      if (i == 4 && j == 2) {
        expect_complex_near(h_B(i, j, 0) / T(Nx * Ny, 0), T(0, -0.5), max_err);
      } else {
        expect_complex_near(h_B(i, j, 0), T(0, 0), max_err);
      }
    }
  }

  // test roundtripping data, with normalization
  plan.inverse(d_B, d_A2);
  gt::copy(d_A2, h_A2);
  for (int i = 0; i < h_A.shape(0); i++) {
    for (int j = 0; j < h_A.shape(1); j++) {
      ASSERT_NEAR(h_A(i, j, 0), h_A2(i, j, 0) / (Nx * Ny), max_err);
    }
  }
}

TEST(fft, r2c_2d)
{
  fft_r2c_2d<float>();
}

TEST(fft, d2z_2d)
{
  fft_r2c_2d<double>();
}

template <typename E>
void fft_c2c_2d()
{
  constexpr int Nx = 17;
  constexpr int Ny = 5;
  constexpr int batch_size = 1;
  using T = gt::complex<E>;

  auto h_A = gt::zeros<T>({Nx, Ny, batch_size});
  auto d_A = gt::empty_device<T>(h_A.shape());

  auto h_A2 = gt::empty<T>(h_A.shape());
  auto d_A2 = gt::empty_device<T>(h_A.shape());

  auto h_B = gt::empty<T>({Nx, Ny, batch_size});
  auto d_B = gt::empty_device<T>(h_B.shape());

  // origin at center of domain has value 1, to model delta function
  h_A(Nx / 2 + 1, Ny / 2 + 1, 0) = T(1.0, 0.0);

  gt::copy(h_A, d_A);

  gt::fft::FFTPlanMany<gt::fft::Domain::COMPLEX, E> plan({Ny, Nx}, batch_size);
  plan(d_A, d_B);
  gt::copy(d_B, h_B);

  // FFT of delta function is all ones in magnitude
  auto h_B_flat = gt::flatten(h_B);
  double max_err = gt::test::detail::max_err<E>::value;
  for (int i = 0; i < h_B_flat.shape(0); i++) {
    ASSERT_NEAR(gt::abs(h_B_flat(i)), 1.0, max_err);
  }

  // test roundtripping data, with normalization
  plan.inverse(d_B, d_A2);
  gt::copy(d_A2, h_A2);
  for (int i = 0; i < h_A.shape(0); i++) {
    for (int j = 0; j < h_A.shape(1); j++) {
      expect_complex_near(h_A(i, j, 0), h_A2(i, j, 0) / T(Nx * Ny, 0));
    }
  }
}

TEST(fft, c2c_2d)
{
  fft_c2c_2d<float>();
}

TEST(fft, z2z_2d)
{
  fft_c2c_2d<double>();
}

template <typename E>
void fft_r2c_3d()
{
  constexpr int Nx = 17;
  constexpr int Ny = 11;
  constexpr int Nz = 5;
  constexpr int batch_size = 1;
  using T = gt::complex<E>;

  auto h_A = gt::zeros<E>({Nx, Ny, Nz, batch_size});
  auto d_A = gt::empty_device<E>(h_A.shape());

  auto h_A2 = gt::empty<E>(h_A.shape());
  auto d_A2 = gt::empty_device<E>(h_A.shape());

  auto h_B = gt::empty<T>({Nx / 2 + 1, Ny, Nz, batch_size});
  auto d_B = gt::empty_device<T>(h_B.shape());

  // origin at center of domain has value 1, to model delta function
  h_A(Nx / 2 + 1, Ny / 2 + 1, Nz / 2 + 1, 0) = 1.0;

  gt::copy(h_A, d_A);

  gt::fft::FFTPlanMany<gt::fft::Domain::REAL, E> plan({Nz, Ny, Nx}, batch_size);
  plan(d_A, d_B);
  gt::copy(d_B, h_B);

  // FFT of delta function is all ones in magnitude
  auto h_B_flat = gt::flatten(h_B);
  double max_err = gt::test::detail::max_err<E>::value;
  for (int i = 0; i < h_B_flat.shape(0); i++) {
    ASSERT_NEAR(gt::abs(h_B_flat(i)), 1.0, max_err);
  }

  // test roundtripping data, with normalization
  plan.inverse(d_B, d_A2);
  gt::copy(d_A2, h_A2);
  for (int i = 0; i < h_A.shape(0); i++) {
    for (int j = 0; j < h_A.shape(1); j++) {
      for (int k = 0; k < h_A.shape(2); k++) {
        ASSERT_NEAR(h_A(i, j, k, 0), h_A2(i, j, k, 0) / (Nx * Ny * Nz),
                    max_err);
      }
    }
  }
}

TEST(fft, r2c_3d)
{
  fft_r2c_3d<float>();
}

TEST(fft, d2z_3d)
{
  fft_r2c_3d<double>();
}
