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

  gt::blas::handle_t* h = gtblas_create();

  f(h, N, a, gt::backend::raw_pointer_cast(d_x.data()), 1,
    gt::backend::raw_pointer_cast(d_y.data()), 1);

  gtblas_destroy(h);

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

  gt::blas::handle_t* h = gtblas_create();

  f(h, N, a, gt::backend::raw_pointer_cast(d_x.data()), 1,
    gt::backend::raw_pointer_cast(d_y.data()), 1);

  gtblas_destroy(h);

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
