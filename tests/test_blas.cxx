#include <gtest/gtest.h>

#include "gtensor/gtensor.h"

#include "gtensor/blas.h"

/*
#define DOUBLE_PREC
#include "gtensor/gpublas.h"
*/

#include "test_debug.h"

template <typename T>
void test_axpy_real()
{
  constexpr int N = 1024;
  gt::gtensor<T, 1> h_x(N);
  gt::gtensor_device<T, 1> d_x(N);
  gt::gtensor<T, 1> h_y(N);
  gt::gtensor_device<T, 1> d_y(N);
  T a = 0.5;

  for (int i = 0; i < N; i++) {
    h_x(i) = 2.0 * static_cast<double>(i);
    h_y(i) = static_cast<double>(i);
  }

  gt::copy(h_x, d_x);
  gt::copy(h_y, d_y);

  gt::blas::handle_t* h = gt::blas::create();

  // gt::blas::axpy(h, N, &a, gt::raw_pointer_cast(d_x.data()), 1,
  //               gt::raw_pointer_cast(d_y.data()), 1);
  gt::blas::axpy(h, a, d_x, d_y);

  gt::blas::destroy(h);

  gt::copy(d_y, h_y);

  for (int i = 0; i < N; i++) {
    EXPECT_EQ(h_y(i), static_cast<T>(i * 2.0));
  }
}

TEST(blas, saxpy)
{
  test_axpy_real<float>();
}

TEST(blas, daxpy)
{
  test_axpy_real<double>();
}

template <typename R>
void test_axpy_complex()
{
  constexpr int N = 1024;
  using T = gt::complex<R>;
  gt::gtensor<T, 1> h_x(N);
  gt::gtensor_device<T, 1> d_x(N);
  gt::gtensor<T, 1> h_y(N);
  gt::gtensor_device<T, 1> d_y(N);
  T a = T(0.5, 0);

  for (int i = 0; i < N; i++) {
    h_x(i) = T(2.0 * i, -2.0 * i);
    h_y(i) = T(1.0 * i, -1.0 * i);
  }

  gt::copy(h_x, d_x);
  gt::copy(h_y, d_y);

  gt::blas::handle_t* h = gt::blas::create();

  gt::blas::axpy(h, N, a, gt::raw_pointer_cast(d_x.data()), 1,
                 gt::raw_pointer_cast(d_y.data()), 1);

  gt::blas::destroy(h);

  gt::copy(d_y, h_y);

  for (int i = 0; i < N; i++) {
    EXPECT_EQ(h_y(i), T(i * 2.0, i * -2.0));
  }
}

TEST(blas, caxpy)
{
  test_axpy_complex<float>();
}

TEST(blas, zaxpy)
{
  test_axpy_complex<double>();
}

template <typename R>
void test_scal_complex()
{
  constexpr int N = 1024;
  using T = gt::complex<R>;
  gt::gtensor<T, 1> h_x(N);
  gt::gtensor<T, 1> h_y(N);
  gt::gtensor_device<T, 1> d_x(N);
  gt::gtensor_device<T, 1> d_y(N);
  T a = T(0.5, 1);
  R b = 0.5;

  for (int i = 0; i < N; i++) {
    h_x(i) = T(2.0 * i, -2.0 * i);
  }

  gt::copy(h_x, d_x);
  gt::copy(h_x, d_y);

  gt::blas::handle_t* h = gt::blas::create();

  // gt::blas::scal(h, N, a, gt::raw_pointer_cast(d_x.data()), 1);
  gt::blas::scal(h, a, d_x);
  gt::blas::scal(h, b, d_y);

  gt::blas::destroy(h);

  gt::copy(d_x, h_x);
  gt::copy(d_y, h_y);

  for (int i = 0; i < N; i++) {
    EXPECT_EQ(h_x(i), T(i * 3.0, i * 1.0));
    EXPECT_EQ(h_y(i), T(i, -1.0 * i));
  }
}

TEST(blas, cscal)
{
  test_scal_complex<float>();
}

TEST(blas, zscal)
{
  test_scal_complex<double>();
}

template <typename T>
void test_scal_real()
{
  constexpr int N = 1024;
  gt::gtensor<T, 1> h_x(N);
  gt::gtensor_device<T, 1> d_x(N);
  T a = T(0.5);

  for (int i = 0; i < N; i++) {
    h_x(i) = T(2.0 * i);
  }

  gt::copy(h_x, d_x);

  gt::blas::handle_t* h = gt::blas::create();

  // gt::blas::scal(h, N, a, gt::raw_pointer_cast(d_x.data()), 1);
  gt::blas::scal(h, a, d_x);

  gt::blas::destroy(h);

  gt::copy(d_x, h_x);

  for (int i = 0; i < N; i++) {
    EXPECT_EQ(h_x(i), T(i * 1.0));
  }
}

TEST(blas, sscal)
{
  test_scal_real<float>();
}

TEST(blas, dscal)
{
  test_scal_real<double>();
}

template <typename R>
void test_copy_complex()
{
  constexpr int N = 1024;
  using T = gt::complex<R>;
  gt::gtensor<T, 1> h_x(N);
  gt::gtensor_device<T, 1> d_x(N);
  gt::gtensor<T, 1> h_y(N);
  gt::gtensor_device<T, 1> d_y(N);

  for (int i = 0; i < N; i++) {
    h_x(i) = T(2.0 * i, -2.0 * i);
    h_y(i) = 0.0;
  }

  gt::copy(h_x, d_x);
  gt::copy(h_y, d_y);

  gt::blas::handle_t* h = gt::blas::create();

  gt::blas::copy(h, d_x, d_y);

  gt::blas::destroy(h);

  gt::copy(d_y, h_y);

  for (int i = 0; i < N; i++) {
    EXPECT_EQ(h_y(i), T(i * 2.0, i * -2.0));
  }
}

TEST(blas, ccopy)
{
  test_copy_complex<float>();
}

TEST(blas, zcopy)
{
  test_copy_complex<double>();
}

template <typename T>
void test_dot_real()
{
  constexpr int N = 16;
  gt::gtensor<T, 1> h_x(N);
  gt::gtensor_device<T, 1> d_x(N);
  gt::gtensor<T, 1> h_y(N);
  gt::gtensor_device<T, 1> d_y(N);

  for (int i = 0; i < N; i++) {
    h_x(i) = T(2.0 * i);
    h_y(i) = T(1.0);
  }

  gt::copy(h_x, d_x);
  gt::copy(h_y, d_y);

  gt::blas::handle_t* h = gt::blas::create();

  T result = gt::blas::dot(h, d_x, d_y);

  gt::blas::destroy(h);

  // sum of first (N-1) integers * 2
  EXPECT_EQ(result, (N - 1) * N);
}

TEST(blas, sdot)
{
  test_dot_real<float>();
}

TEST(blas, ddot)
{
  test_dot_real<double>();
}

template <typename R>
void test_dot_complex()
{
  constexpr int N = 16;
  using T = gt::complex<R>;
  gt::gtensor<T, 1> h_x(N);
  gt::gtensor_device<T, 1> d_x(N);
  gt::gtensor<T, 1> h_y(N);
  gt::gtensor_device<T, 1> d_y(N);

  for (int i = 0; i < N; i++) {
    h_x(i) = T(2.0 * i, -2.0 * i);
    h_y(i) = T(1.0, 0.0);
  }

  gt::copy(h_x, d_x);
  gt::copy(h_y, d_y);

  gt::blas::handle_t* h = gt::blas::create();

  T resultu = gt::blas::dotu(h, d_x, d_y);
  T resultc = gt::blas::dotc(h, d_x, d_y);

  gt::blas::destroy(h);

  // sum of first (N-1) integers * 2, which is the sum of real values
  // of x
  R sum = (N - 1) * N;
  EXPECT_EQ(resultu, T(sum, -1.0 * sum));
  EXPECT_EQ(resultc, T(sum, sum));
}

TEST(blas, cdot)
{
  test_dot_complex<float>();
}

TEST(blas, zdot)
{
  test_dot_complex<double>();
}

template <typename T>
void test_gemv_real()
{
  constexpr int N = 16;
  gt::gtensor<T, 2> h_A(gt::shape(N, N));
  gt::gtensor_device<T, 2> d_A(gt::shape(N, N));
  gt::gtensor<T, 1> h_x(N);
  gt::gtensor_device<T, 1> d_x(N);
  gt::gtensor<T, 1> h_y(N);
  gt::gtensor_device<T, 1> d_y(N);
  T a = 0.5;
  T b = 2.0;

  for (int i = 0; i < N; i++) {
    h_x(i) = 2.0;
    h_y(i) = i / 2.0;
    for (int j = 0; j < N; j++) {
      h_A(j, i) = static_cast<double>(i + j * N);
    }
  }

  /*
   * a * x = {1, 1, 1, 1, ...}
   * b * y = {0, 1, 2, 3, ...}
   * M     = {{0, 1, 2, 3, ...},
   *          {N, N+1, N+2, ...},
   *          {2N, 2N+1, ...}
   *          ...
   *          {(N-1)*N, (N-1)*N + 1, ...}}
   *
   * julia code
   *
   * N = 16;
   * a = 0.5;
   * b = 2.0;
   * x = fill(2.0, N);
   * y = collect(0:N-1) / 2.0;
   * M = [Float64(j+i*N) for i=0:N-1, j=0:N-1]
   * y = a * M * x + b * y
   */

  gt::copy(h_A, d_A);
  gt::copy(h_x, d_x);
  gt::copy(h_y, d_y);

  gt::blas::handle_t* h = gt::blas::create();

  // gt::blas::gemv(h, N, N, a, gt::raw_pointer_cast(d_A.data()), N,
  //                gt::raw_pointer_cast(d_x.data()), 1, b,
  //                gt::raw_pointer_cast(d_y.data()), 1);
  gt::blas::gemv(h, a, d_A, d_x, b, d_y);

  gt::blas::destroy(h);

  gt::copy(d_y, h_y);

  double row_sum = 0.0;
  for (int i = 0; i < N; i++) {
    // Since a*x is just ones, a*M*x computes row sums.
    // Based on construction of M, row sums can be computed by
    // calculating difference of triangle numbers - first row is triangle
    // number (N-1), T_{N-1} = (N-1)*N/2.0. Second row is
    // T_{2N-1} - T_{N-1}, third row is T_{3N-1} - T_{2N-1}, etc
    row_sum = ((i + 1) * N * ((i + 1) * N - 1) - i * N * (i * N - 1)) / 2.0;

    // b*y is just i, so total is (row_sum + i)
    EXPECT_EQ(h_y(i), T(row_sum + i));
  }
}

TEST(blas, sgemv)
{
  test_gemv_real<float>();
}

TEST(blas, dgemv)
{
  test_gemv_real<double>();
}

template <typename R>
void test_gemv_complex()
{
  constexpr int N = 32;
  using T = gt::complex<R>;
  gt::gtensor<T, 1> h_x(N);
  gt::gtensor_device<T, 1> d_x(N);
  gt::gtensor<T, 1> h_y(N);
  gt::gtensor_device<T, 1> d_y(N);
  gt::gtensor<T, 2> h_mat(gt::shape(N, N));
  gt::gtensor_device<T, 2> d_mat(gt::shape(N, N));
  T a = T(0.5, 1.0);
  T b = T(-1.0, 2.0);

  for (int i = 0; i < N; i++) {
    h_x(i) = T(i, 0.0);
    h_y(i) = T(0.0, i);
    for (int j = 0; j < N; ++j) {
      h_mat(i, j) = T(i, j);
    }
  }

  gt::copy(h_x, d_x);
  gt::copy(h_y, d_y);
  gt::copy(h_mat, d_mat);

  gt::blas::handle_t* h = gt::blas::create();

  // gt::blas::gemv(h, N, N, a, gt::raw_pointer_cast(d_mat.data()), N,
  //                gt::raw_pointer_cast(d_x.data()), 1, b,
  //                gt::raw_pointer_cast(d_y.data()), 1);
  gt::blas::gemv(h, a, d_mat, d_x, b, d_y);

  gt::blas::destroy(h);

  gt::copy(d_y, h_y);

  for (int p = 0; p < N; p++) {
    auto r = p * (N * (N + 1) / 2 - N);
    auto s = (N * (N + 1) * (2 * N + 1) - 6 * N * N) / 6;
    EXPECT_EQ(h_y(p), T(a.real() * r - a.imag() * s - b.imag() * p,
                        a.imag() * r + a.real() * s + b.real() * p));
  }
}

TEST(blas, cgemv)
{
  test_gemv_complex<float>();
}

TEST(blas, zgemv)
{
  test_gemv_complex<double>();
}
