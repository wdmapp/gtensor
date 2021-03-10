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

  f(N, &a, gt::backend::raw_pointer_cast(d_x.data()), 1,
    gt::backend::raw_pointer_cast(d_y.data()), 1);

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

  f(N, &a, gt::backend::raw_pointer_cast(d_x.data()), 1,
    gt::backend::raw_pointer_cast(d_y.data()), 1);

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

  f(N, &a, gt::backend::raw_pointer_cast(d_x.data()), 1);

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

  f(N, &a, gt::backend::raw_pointer_cast(d_x.data()), 1);
  f2(N, &a2, gt::backend::raw_pointer_cast(d_y.data()), 1);

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
