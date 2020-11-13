#include <gtest/gtest.h>

#include "gtensor/gtensor.h"

#include "gtensor/gpufft.h"

#include "test_helpers.h"

TEST(fft, d2z_1d)
{
  constexpr int N = 4;
  constexpr int Nout = N / 2 + 1;
  constexpr int RANK = 1;
  constexpr int batch_size = 2;
  using E = double;
  using T = gt::complex<E>;
  gpufft_handle_t plan;
  int lengths[RANK] = {N};

  E* h_A = gt::backend::host_allocator<E>::allocate(batch_size * N);
  E* d_A = gt::backend::device_allocator<E>::allocate(batch_size * N);

  T* h_B = gt::backend::host_allocator<T>::allocate(batch_size * Nout);
  T* d_B = gt::backend::device_allocator<T>::allocate(batch_size * Nout);

  // x = [2 3 -1 4];
  h_A[0] = 2;
  h_A[1] = 3;
  h_A[2] = -1;
  h_A[3] = 4;

  // y = [7 -21 11 1];
  h_A[4] = 7;
  h_A[5] = -21;
  h_A[6] = 11;
  h_A[7] = 1;

  // zero output array, rocfft at least does not zero padding elements
  // for real to complex transform
  gt::backend::device_memset(d_B, 0, batch_size * Nout * sizeof(*d_B));

  gt::backend::device_copy_hd(h_A, d_A, batch_size * N);

  // fft(x) -> [8+0i 3+1i -6+0i 3-1i]
  // but with fftw convention for real transforms, the last term is
  // conjugate of second and set to 0 to save storage / computation
  gpufft_plan_many(&plan, 1, lengths, 1, N, 1, Nout, GPUFFT_D2Z, batch_size);
  gpufft_exec_d2z(plan, d_A, (gpufft_double_complex_t*)d_B);
  gpufft_plan_destroy(plan);

  gt::backend::device_copy_dh(d_B, h_B, batch_size * Nout);

  expect_complex_eq(h_B[0], T(8, 0));
  expect_complex_eq(h_B[1], T(3, 1));
  expect_complex_eq(h_B[2], T(-6, 0));

  expect_complex_eq(h_B[3], T(-2, 0));
  expect_complex_eq(h_B[4], T(-4, 22));
  expect_complex_eq(h_B[5], T(38, 0));

  gt::backend::host_allocator<E>::deallocate(h_A);
  gt::backend::device_allocator<E>::deallocate(d_A);

  gt::backend::host_allocator<T>::deallocate(h_B);
  gt::backend::device_allocator<T>::deallocate(d_B);
}

TEST(fft, z2d_1d)
{
  constexpr int N = 4;
  constexpr int Ncomplex = N / 2 + 1;
  constexpr int RANK = 1;
  constexpr int batch_size = 2;
  using E = double;
  using T = gt::complex<E>;
  gpufft_handle_t plan;
  int lengths[RANK] = {N};

  E* h_A = gt::backend::host_allocator<E>::allocate(batch_size * N);
  E* d_A = gt::backend::device_allocator<E>::allocate(batch_size * N);

  T* h_B = gt::backend::host_allocator<T>::allocate(batch_size * Ncomplex);
  T* d_B = gt::backend::device_allocator<T>::allocate(batch_size * Ncomplex);

  h_B[0] = T(8, 0);
  h_B[1] = T(3, 1);
  h_B[2] = T(-6, 0);

  h_B[3] = T(-2, 0);
  h_B[4] = T(-4, 22);
  h_B[5] = T(38, 0);

  gt::backend::device_copy_hd(h_B, d_B, batch_size * Ncomplex);

  // ifft(x) -> [8+0i 3+1i -6+0i 3-1i]
  gpufft_plan_many(&plan, 1, lengths, 1, Ncomplex, 1, N, GPUFFT_Z2D,
                   batch_size);
  gpufft_exec_z2d(plan, (gpufft_double_complex_t*)d_B, d_A);
  gpufft_plan_destroy(plan);

  gt::backend::device_copy_dh(d_A, h_A, batch_size * N);

  EXPECT_EQ(h_A[0] / N, 2);
  EXPECT_EQ(h_A[1] / N, 3);
  EXPECT_EQ(h_A[2] / N, -1);
  EXPECT_EQ(h_A[3] / N, 4);

  EXPECT_EQ(h_A[4] / N, 7);
  EXPECT_EQ(h_A[5] / N, -21);
  EXPECT_EQ(h_A[6] / N, 11);
  EXPECT_EQ(h_A[7] / N, 1);

  gt::backend::host_allocator<E>::deallocate(h_A);
  gt::backend::device_allocator<E>::deallocate(d_A);

  gt::backend::host_allocator<T>::deallocate(h_B);
  gt::backend::device_allocator<T>::deallocate(d_B);
}
