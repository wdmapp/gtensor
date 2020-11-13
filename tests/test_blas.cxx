#include <gtest/gtest.h>

#include "gtensor/gtensor.h"

#define DOUBLE_PREC
#include "gtensor/gpublas.h"

TEST(axpy, daxpy)
{
  constexpr int N = 1024;
  using T = double;
  T* h_x = gt::backend::host_allocator<T>::allocate(N);
  T* d_x = gt::backend::device_allocator<T>::allocate(N);
  T* h_y = gt::backend::host_allocator<T>::allocate(N);
  T* d_y = gt::backend::device_allocator<T>::allocate(N);
  T a = 0.5;

  for (int i = 0; i < N; i++) {
    h_x[i] = 2.0 * static_cast<double>(i);
    h_y[i] = static_cast<double>(i);
  }

  gt::backend::device_copy_hd(h_x, d_x, N);
  gt::backend::device_copy_hd(h_y, d_y, N);

  gpublas_create();

  gpublas_daxpy(N, &a, d_x, 1, d_y, 1);

  gpublas_destroy();

  gt::backend::device_copy_dh(d_y, h_y, N);

  for (int i = 0; i < N; i++) {
    EXPECT_EQ(h_y[i], static_cast<T>(i * 2.0));
  }

  gt::backend::host_allocator<T>::deallocate(h_x);
  gt::backend::device_allocator<T>::deallocate(d_x);
  gt::backend::host_allocator<T>::deallocate(h_y);
  gt::backend::device_allocator<T>::deallocate(d_y);
}

TEST(axpy, zaxpy)
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

  gpublas_create();

  gpublas_zaxpy(N, (gpublas_complex_double_t*)&a,
                (gpublas_complex_double_t*)(d_x), 1,
                (gpublas_complex_double_t*)(d_y), 1);

  gpublas_destroy();

  gt::backend::device_copy_dh(d_y, h_y, N);

  for (int i = 0; i < N; i++) {
    EXPECT_EQ(h_y[i], T(i * 2.0, i * -2.0));
  }

  gt::backend::host_allocator<T>::deallocate(h_x);
  gt::backend::device_allocator<T>::deallocate(d_x);
  gt::backend::host_allocator<T>::deallocate(h_y);
  gt::backend::device_allocator<T>::deallocate(d_y);
}

TEST(scal, zdscal)
{
  constexpr int N = 1024;
  using T = gt::complex<double>;
  T* h_x = gt::backend::host_allocator<T>::allocate(N);
  T* d_x = gt::backend::device_allocator<T>::allocate(N);
  double a = 0.5;

  for (int i = 0; i < N; i++) {
    h_x[i] = T(2.0 * i, -2.0 * i);
  }

  gt::backend::device_copy_hd(h_x, d_x, N);

  gpublas_create();

  gpublas_zdscal(N, a, (gpublas_complex_double_t*)(d_x), 1);

  gpublas_destroy();

  gt::backend::device_copy_dh(d_x, h_x, N);

  for (int i = 0; i < N; i++) {
    EXPECT_EQ(h_x[i], T(i * 1.0, i * -1.0));
  }

  gt::backend::host_allocator<T>::deallocate(h_x);
  gt::backend::device_allocator<T>::deallocate(d_x);
}

TEST(copy, zcopy)
{
  constexpr int N = 1024;
  using T = gt::complex<double>;
  T* h_x = gt::backend::host_allocator<T>::allocate(N);
  T* d_x = gt::backend::device_allocator<T>::allocate(N);
  T* h_y = gt::backend::host_allocator<T>::allocate(N);
  T* d_y = gt::backend::device_allocator<T>::allocate(N);

  for (int i = 0; i < N; i++) {
    h_x[i] = T(2.0 * i, -2.0 * i);
    h_y[i] = 0.0;
  }

  gt::backend::device_copy_hd(h_x, d_x, N);
  gt::backend::device_copy_hd(h_y, d_y, N);

  gpublas_create();

  gpublas_zcopy(N, (gpublas_complex_double_t*)(d_x), 1,
                (gpublas_complex_double_t*)(d_y), 1);

  gpublas_destroy();

  gt::backend::device_copy_dh(d_y, h_y, N);

  for (int i = 0; i < N; i++) {
    EXPECT_EQ(h_y[i], T(i * 2.0, i * -2.0));
  }

  gt::backend::host_allocator<T>::deallocate(h_x);
  gt::backend::device_allocator<T>::deallocate(d_x);
  gt::backend::host_allocator<T>::deallocate(h_y);
  gt::backend::device_allocator<T>::deallocate(d_y);
}


