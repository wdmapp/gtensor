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

#if 0
TEST(blas, zaxpy)
{
  constexpr int N = 1024;
  using T = gt::complex<double>;
  T* h_x = gt::backend::host_allocator<T>::allocate(N);
  T* d_x = gt::backend::device_allocator<T>::allocate(N);
  T* h_y = gt::backend::host_allocator<T>::allocate(N);
  T* d_y = gt::backend::device_allocator<T>::allocate(N);
  T a = T(0.5, 0);

  for (int i = 0; i < N; i++) {
    h_x[i] = T(2.0 * i, -2.0 * i);
    h_y[i] = T(1.0 * i, -1.0 * i);
  }

  gt::backend::device_copy_hd(h_x, d_x, N);
  gt::backend::device_copy_hd(h_y, d_y, N);

  gtblas_create();

  gtblas_zaxpy(N, (gtblas_complex_double_t*)&a, (gtblas_complex_double_t*)(d_x),
               1, (gtblas_complex_double_t*)(d_y), 1);

  gtblas_destroy();

  gt::backend::device_copy_dh(d_y, h_y, N);

  for (int i = 0; i < N; i++) {
    EXPECT_EQ(h_y[i], T(i * 2.0, i * -2.0));
  }

  gt::backend::host_allocator<T>::deallocate(h_x);
  gt::backend::device_allocator<T>::deallocate(d_x);
  gt::backend::host_allocator<T>::deallocate(h_y);
  gt::backend::device_allocator<T>::deallocate(d_y);
}

#endif
