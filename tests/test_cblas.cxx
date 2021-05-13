#include <gtest/gtest.h>

#include "gtensor/gtensor.h"

#include "gtensor/cblas.h"

template <typename T, typename F>
void test_real_axpy(F&& f)
{
  constexpr int N = 1024;
  gt::gtensor<T, 1> h_x(N);
  gt::gtensor_device<T, 1> d_x(N);
  gt::gtensor<T, 1> h_y(N);
  gt::gtensor_device<T, 1> d_y(N);
  T a = 0.5;

  for (int i = 0; i < N; i++) {
    h_x(i) = 2.0 * T(i);
    h_y(i) = T(i);
  }

  gt::copy(h_x, d_x);
  gt::copy(h_y, d_y);

  gtblas_create();

  f(N, &a, gt::raw_pointer_cast(d_x.data()), 1,
    gt::raw_pointer_cast(d_y.data()), 1);

  gtblas_destroy();

  gt::copy(d_y, h_y);

  for (int i = 0; i < N; i++) {
    EXPECT_EQ(h_y(i), T(i * 2.0));
  }
}

TEST(cblas, saxpy)
{
  test_real_axpy<float>(&gtblas_saxpy);
}

TEST(cblas, daxpy)
{
  test_real_axpy<double>(&gtblas_daxpy);
}

template <typename R, typename F>
void test_complex_axpy(F&& f)
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

  gtblas_create();

  f(N, &a, gt::raw_pointer_cast(d_x.data()), 1,
    gt::raw_pointer_cast(d_y.data()), 1);

  gtblas_destroy();

  gt::copy(d_y, h_y);

  for (int i = 0; i < N; i++) {
    EXPECT_EQ(h_y(i), T(i * 2.0, i * -2.0));
  }
}

TEST(cblas, caxpy)
{
  test_complex_axpy<float>(&gtblas_caxpy);
}

TEST(cblas, zaxpy)
{
  test_complex_axpy<double>(&gtblas_zaxpy);
}

template <typename T, typename F>
void test_real_scal(F&& f)
{
  constexpr int N = 1024;
  gt::gtensor<T, 1> h_x(N);
  gt::gtensor_device<T, 1> d_x(N);
  T a = 0.5;

  for (int i = 0; i < N; i++) {
    h_x(i) = 2.0 * T(i);
  }

  gt::copy(h_x, d_x);

  gtblas_create();

  f(N, &a, gt::raw_pointer_cast(d_x.data()), 1);

  gtblas_destroy();

  gt::copy(d_x, h_x);

  for (int i = 0; i < N; i++) {
    EXPECT_EQ(h_x(i), T(i));
  }
}

TEST(cblas, sscal)
{
  test_real_scal<float>(&gtblas_sscal);
}

TEST(cblas, dscal)
{
  test_real_scal<double>(&gtblas_dscal);
}

template <typename R, typename F, typename F2>
void test_complex_scal(F&& f, F2&& f2)
{
  constexpr int N = 1024;
  using T = gt::complex<R>;
  gt::gtensor<T, 1> h_x(N);
  gt::gtensor_device<T, 1> d_x(N);
  gt::gtensor<T, 1> h_y(N);
  gt::gtensor_device<T, 1> d_y(N);
  T a = T(0.5, 1);
  R a2 = 0.5;

  for (int i = 0; i < N; i++) {
    h_x(i) = T(2.0 * i, -2.0 * i);
  }

  gt::copy(h_x, d_x);
  gt::copy(h_x, d_y);

  gtblas_create();

  f(N, &a, gt::raw_pointer_cast(d_x.data()), 1);
  f2(N, &a2, gt::raw_pointer_cast(d_y.data()), 1);

  gtblas_destroy();

  gt::copy(d_x, h_x);
  gt::copy(d_y, h_y);

  for (int i = 0; i < N; i++) {
    EXPECT_EQ(h_x(i), T(i * 3.0, i * 1.0));
    EXPECT_EQ(h_y(i), T(i, i * -1.0));
  }
}

TEST(cblas, cscal)
{
  test_complex_scal<float>(&gtblas_cscal, &gtblas_csscal);
}

TEST(cblas, zscal)
{
  test_complex_scal<double>(&gtblas_zscal, &gtblas_zdscal);
}

template <typename T, typename F>
void test_gemv_real(F&& f)
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

  gtblas_create();

  f(N, N, &a, gt::raw_pointer_cast(d_A.data()), N,
    gt::raw_pointer_cast(d_x.data()), 1, &b, gt::raw_pointer_cast(d_y.data()),
    1);

  gtblas_destroy();

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

TEST(cblas, sgemv)
{
  test_gemv_real<float>(&gtblas_sgemv);
}

TEST(cblas, dgemv)
{
  test_gemv_real<double>(&gtblas_dgemv);
}

template <typename R, typename F>
void test_gemv_complex(F&& f)
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

  gtblas_create();

  f(N, N, &a, gt::raw_pointer_cast(d_mat.data()), N,
    gt::raw_pointer_cast(d_x.data()), 1, &b, gt::raw_pointer_cast(d_y.data()),
    1);

  gtblas_destroy();

  gt::copy(d_y, h_y);

  for (int p = 0; p < N; p++) {
    auto r = p * (N * (N + 1) / 2 - N);
    auto s = (N * (N + 1) * (2 * N + 1) - 6 * N * N) / 6;
    EXPECT_EQ(h_y(p), T(a.real() * r - a.imag() * s - b.imag() * p,
                        a.imag() * r + a.real() * s + b.real() * p));
  }
}

TEST(cblas, cgemv)
{
  test_gemv_complex<float>(&gtblas_cgemv);
}

TEST(cblas, zgemv)
{
  test_gemv_complex<double>(&gtblas_zgemv);
}
