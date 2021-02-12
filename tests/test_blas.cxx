#include <gtest/gtest.h>

#include "gtensor/gtensor.h"

#define DOUBLE_PREC
#include "gtensor/gpublas.h"

#include "test_debug.h"

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

TEST(axpy, dgemv)
{
  constexpr int N = 16;
  using T = double;
  T* h_A = gt::backend::host_allocator<T>::allocate(N * N);
  T* d_A = gt::backend::device_allocator<T>::allocate(N * N);
  T* h_x = gt::backend::host_allocator<T>::allocate(N);
  T* d_x = gt::backend::device_allocator<T>::allocate(N);
  T* h_y = gt::backend::host_allocator<T>::allocate(N);
  T* d_y = gt::backend::device_allocator<T>::allocate(N);
  T a = 0.5;
  T b = 2.0;

  for (int i = 0; i < N; i++) {
    h_x[i] = 2.0;
    h_y[i] = i / 2.0;
    for (int j = 0; j < N; j++) {
      h_A[j + i * N] = static_cast<double>(i + j * N);
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

  gt::backend::device_copy_hd(h_A, d_A, N * N);
  gt::backend::device_copy_hd(h_x, d_x, N);
  gt::backend::device_copy_hd(h_y, d_y, N);

  gpublas_create();

  gpublas_dgemv(N, N, &a, d_A, N, d_x, 1, &b, d_y, 1);

  gpublas_destroy();

  gt::backend::device_copy_dh(d_y, h_y, N);

  double row_sum = 0.0;
  for (int i = 0; i < N; i++) {
    // Since a*x is just ones, a*M*x computes row sums.
    // Based on construction of M, row sums can be computed by
    // calculating difference of triangle numbers - first row is triangle
    // number (N-1), T_{N-1} = (N-1)*N/2.0. Second row is
    // T_{2N-1} - T_{N-1}, third row is T_{3N-1} - T_{2N-1}, etc
    row_sum = ((i + 1) * N * ((i + 1) * N - 1) - i * N * (i * N - 1)) / 2.0;

    // b*y is just i, so total is (row_sum + i)
    EXPECT_EQ(h_y[i], T(row_sum + i));
  }

  gt::backend::host_allocator<T>::deallocate(h_A);
  gt::backend::device_allocator<T>::deallocate(d_A);
  gt::backend::host_allocator<T>::deallocate(h_x);
  gt::backend::device_allocator<T>::deallocate(d_x);
  gt::backend::host_allocator<T>::deallocate(h_y);
  gt::backend::device_allocator<T>::deallocate(d_y);
}
