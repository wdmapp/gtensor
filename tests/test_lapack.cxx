#include <gtest/gtest.h>

#include "gtensor/gtensor.h"

#include "gtensor/blas.h"

#include "test_helpers.h"

template <typename T>
void set_A(T* h_A)
{
  // matlab/octave:
  //  A = [1 2 2; 4 4 2; 4 6 4];
  //  L,U,p = lu(A)
  // first column
  h_A[0] = 1;
  h_A[1] = 4;
  h_A[2] = 4;
  // second column
  h_A[3] = 2;
  h_A[4] = 4;
  h_A[5] = 6;
  // third column
  h_A[6] = 2;
  h_A[7] = 2;
  h_A[8] = 4;
}

template <typename T>
void set_A_LU(T* h_A)
{
  // first column factored
  h_A[0] = 4.0;
  h_A[1] = 1.0;
  h_A[2] = 0.25;
  // second column
  h_A[3] = 4.0;
  h_A[4] = 2.0;
  h_A[5] = 0.5;
  // thrid column
  h_A[6] = 2.0;
  h_A[7] = 2.0;
  h_A[8] = 0.5;
}

template <typename T>
void set_A_piv(T* h_p)
{
  h_p[0] = 2;
  h_p[1] = 3;
  h_p[2] = 3;
}

template <typename T>
void set_B_complex(T* h_B)
{
  // second matrix, complex
  // matlab/octave:
  //  B = [1+i 2-i 2; 4i 4 2; 4 6i 4];
  //  L,U,p = lu(A2);
  // first column
  h_B[0] = T(1, 1);
  h_B[1] = T(0, 4);
  h_B[2] = T(4, 0);
  // second column
  h_B[3] = T(2, -1);
  h_B[4] = T(4, 0);
  h_B[5] = T(0, 6);
  // third column
  h_B[6] = T(2, 0);
  h_B[7] = T(2, 0);
  h_B[8] = T(4, 0);
}

template <typename T>
void set_B_LU(T* h_B)
{
  // first column
  h_B[0] = T(0, 4);
  h_B[1] = T(0, -1);
  h_B[2] = T(0.25, -0.25);
  // second column factored
  h_B[3] = T(4, 0);
  h_B[4] = T(0, 10);
  h_B[5] = T(0, -0.1);
  // third column factored
  h_B[6] = T(2, 0);
  h_B[7] = T(4, 2);
  h_B[8] = T(1.3, 0.9);
}

template <typename T>
void set_B_piv(T* h_p)
{
  h_p[0] = 2;
  h_p[1] = 3;
  h_p[2] = 3;
}

TEST(getrfs, dgetrf_batch1)
{
  constexpr int N = 3;
  constexpr int S = N * N;
  constexpr int batch_size = 1;
  using T = double;

  T** h_Aptr = gt::backend::host_allocator<T*>::allocate(batch_size);
  T** d_Aptr = gt::backend::device_allocator<T*>::allocate(batch_size);
  T* h_A = gt::backend::host_allocator<T>::allocate(batch_size * S);
  T* d_A = gt::backend::device_allocator<T>::allocate(batch_size * S);

  auto h_p =
    gt::backend::host_allocator<gtblas_index_t>::allocate(batch_size * N);
  auto d_p =
    gt::backend::device_allocator<gtblas_index_t>::allocate(batch_size * N);
  int* h_info = gt::backend::host_allocator<int>::allocate(batch_size);
  int* d_info = gt::backend::device_allocator<int>::allocate(batch_size);

  set_A(h_A);
  h_Aptr[0] = &d_A[0];

  gt::backend::device_copy_hd(h_A, d_A, batch_size * S);
  gt::backend::device_copy_hd(h_Aptr, d_Aptr, batch_size);

  gtblas_create();

  gtblas_dgetrf_batched(N, d_Aptr, N, d_p, d_info, batch_size);

  gtblas_destroy();

  gt::backend::device_copy_dh(d_A, h_A, batch_size * S);
  gt::backend::device_copy_dh(d_p, h_p, batch_size * N);
  gt::backend::device_copy_dh(d_info, h_info, batch_size);

  // first column factored
  EXPECT_EQ(h_A[0], 4.0);
  EXPECT_EQ(h_A[1], 1.0);
  EXPECT_EQ(h_A[2], 0.25);
  // second column factored
  EXPECT_EQ(h_A[3], 4.0);
  EXPECT_EQ(h_A[4], 2.0);
  EXPECT_EQ(h_A[5], 0.5);
  // third column factored
  EXPECT_EQ(h_A[6], 2.0);
  EXPECT_EQ(h_A[7], 2.0);
  EXPECT_EQ(h_A[8], 0.5);

  // Math notation is 2, 3, 1, but BLAS uses procedural notation, i.e.
  // on the thirst step no swap is done so one-based index of third row
  // is still 3 (no swapping).
  EXPECT_EQ(h_p[0], 2);
  EXPECT_EQ(h_p[1], 3);
  EXPECT_EQ(h_p[2], 3);

  for (int b = 0; b < batch_size; b++) {
    // A_i factored successfully
    EXPECT_EQ(h_info[b], 0);
  }

  gt::backend::host_allocator<T*>::deallocate(h_Aptr);
  gt::backend::device_allocator<T*>::deallocate(d_Aptr);
  gt::backend::host_allocator<T>::deallocate(h_A);
  gt::backend::device_allocator<T>::deallocate(d_A);

  gt::backend::host_allocator<gtblas_index_t>::deallocate(h_p);
  gt::backend::device_allocator<gtblas_index_t>::deallocate(d_p);
  gt::backend::host_allocator<int>::deallocate(h_info);
  gt::backend::device_allocator<int>::deallocate(d_info);
}

TEST(getrfs, dgetrs_batch1)
{
  constexpr int N = 3;
  constexpr int NRHS = 2;
  constexpr int S = N * N;
  constexpr int batch_size = 1;
  using T = double;

  T** h_Aptr = gt::backend::host_allocator<T*>::allocate(batch_size);
  T** d_Aptr = gt::backend::device_allocator<T*>::allocate(batch_size);
  T* h_A = gt::backend::host_allocator<T>::allocate(batch_size * S);
  T* d_A = gt::backend::device_allocator<T>::allocate(batch_size * S);

  T** h_Bptr = gt::backend::host_allocator<T*>::allocate(batch_size);
  T** d_Bptr = gt::backend::device_allocator<T*>::allocate(batch_size);
  T* h_B = gt::backend::host_allocator<T>::allocate(batch_size * N * NRHS);
  T* d_B = gt::backend::device_allocator<T>::allocate(batch_size * N * NRHS);

  auto h_p =
    gt::backend::host_allocator<gtblas_index_t>::allocate(batch_size * N);
  auto d_p =
    gt::backend::device_allocator<gtblas_index_t>::allocate(batch_size * N);

  set_A_LU(h_A);
  h_Aptr[0] = &d_A[0];
  set_A_piv(h_p);

  // col vector [11; 18; 28]
  h_B[0] = 11;
  h_B[1] = 18;
  h_B[2] = 28;
  // col vector [73; 78; 154]
  h_B[3] = 73;
  h_B[4] = 78;
  h_B[5] = 154;
  h_Bptr[0] = &d_B[0];
  h_Bptr[1] = &d_B[N];

  gt::backend::device_copy_hd(h_Aptr, d_Aptr, batch_size);
  gt::backend::device_copy_hd(h_A, d_A, batch_size * S);
  gt::backend::device_copy_hd(h_Bptr, d_Bptr, batch_size);
  gt::backend::device_copy_hd(h_B, d_B, batch_size * N * NRHS);
  gt::backend::device_copy_hd(h_B, d_B, batch_size * N * NRHS);
  gt::backend::device_copy_hd(h_p, d_p, batch_size * N);

  gtblas_create();

  gtblas_dgetrs_batched(N, NRHS, d_Aptr, N, d_p, d_Bptr, N, batch_size);

  gtblas_destroy();

  gt::backend::device_copy_dh(d_B, h_B, batch_size * N * NRHS);

  // solution vector [1; 2; 3]
  EXPECT_EQ(h_B[0], 1.0);
  EXPECT_EQ(h_B[1], 2.0);
  EXPECT_EQ(h_B[2], 3.0);
  // solution vector [-3; 7; 31]
  EXPECT_EQ(h_B[3], -3.0);
  EXPECT_EQ(h_B[4], 7.0);
  EXPECT_EQ(h_B[5], 31.0);

  gt::backend::host_allocator<T*>::deallocate(h_Aptr);
  gt::backend::device_allocator<T*>::deallocate(d_Aptr);
  gt::backend::host_allocator<T>::deallocate(h_A);
  gt::backend::device_allocator<T>::deallocate(d_A);

  gt::backend::host_allocator<T>::deallocate(h_B);
  gt::backend::device_allocator<T>::deallocate(d_B);
  gt::backend::host_allocator<T*>::deallocate(h_Bptr);
  gt::backend::device_allocator<T*>::deallocate(d_Bptr);

  gt::backend::host_allocator<gtblas_index_t>::deallocate(h_p);
  gt::backend::device_allocator<gtblas_index_t>::deallocate(d_p);
}

TEST(getrfs, zgetrf_batch2)
{
  constexpr int N = 3;
  constexpr int S = N * N;
  constexpr int batch_size = 2;
  using T = gt::complex<double>;

  T** h_Aptr = gt::backend::host_allocator<T*>::allocate(batch_size);
  T** d_Aptr = gt::backend::device_allocator<T*>::allocate(batch_size);
  T* h_A = gt::backend::host_allocator<T>::allocate(batch_size * S);
  T* d_A = gt::backend::device_allocator<T>::allocate(batch_size * S);

  auto h_p =
    gt::backend::host_allocator<gtblas_index_t>::allocate(batch_size * N);
  auto d_p =
    gt::backend::device_allocator<gtblas_index_t>::allocate(batch_size * N);
  int* h_info = gt::backend::host_allocator<int>::allocate(batch_size);
  int* d_info = gt::backend::device_allocator<int>::allocate(batch_size);

  set_A(h_A);
  set_B_complex(h_A + S);

  h_Aptr[0] = &d_A[0];
  h_Aptr[1] = &d_A[S];

  gt::backend::device_copy_hd(h_A, d_A, batch_size * S);
  gt::backend::device_copy_hd(h_Aptr, d_Aptr, batch_size);

  gtblas_create();

  gtblas_zgetrf_batched(N, (gtblas_complex_double_t**)d_Aptr, N, d_p, d_info,
                         batch_size);

  gtblas_destroy();

  gt::backend::device_copy_dh(d_A, h_A, batch_size * S);
  gt::backend::device_copy_dh(d_p, h_p, batch_size * N);
  gt::backend::device_copy_dh(d_info, h_info, batch_size);

  // first matrix result
  // first column factored
  EXPECT_EQ(h_A[0], 4.0);
  EXPECT_EQ(h_A[1], 1.0);
  EXPECT_EQ(h_A[2], 0.25);
  // second column factored
  EXPECT_EQ(h_A[3], 4.0);
  EXPECT_EQ(h_A[4], 2.0);
  EXPECT_EQ(h_A[5], 0.5);
  // third column factored
  EXPECT_EQ(h_A[6], 2.0);
  EXPECT_EQ(h_A[7], 2.0);
  EXPECT_EQ(h_A[8], 0.5);

  // Math notation is 2, 3, 1, but BLAS uses procedural notation, i.e.
  // on the thirst step no swap is done so one-based index of third row
  // is still 3 (no swapping).
  EXPECT_EQ(h_p[0], 2);
  EXPECT_EQ(h_p[1], 3);
  EXPECT_EQ(h_p[2], 3);

  // second matrix result
  // first column factored
  EXPECT_EQ(h_A[9], T(0, 4));
  EXPECT_EQ(h_A[10], T(0, -1));
  EXPECT_EQ(h_A[11], T(0.25, -0.25));
  // second column factored
  EXPECT_EQ(h_A[12], T(4, 0));
  EXPECT_EQ(h_A[13], T(0, 10));
  EXPECT_EQ(h_A[14], T(0, -0.1));
  // third column factored
  EXPECT_EQ(h_A[15], T(2, 0));
  EXPECT_EQ(h_A[16], T(4, 2));
  EXPECT_EQ(h_A[17], T(1.3, 0.9));

  // Math notation is 2, 3, 1, but BLAS uses procedural notation, i.e.
  // on the thirst step no swap is done so one-based index of third row
  // is still 3 (no swapping).
  EXPECT_EQ(h_p[0], 2);
  EXPECT_EQ(h_p[1], 3);
  EXPECT_EQ(h_p[2], 3);

  for (int b = 0; b < batch_size; b++) {
    // A_i factored successfully
    EXPECT_EQ(h_info[b], 0);
  }

  gt::backend::host_allocator<T*>::deallocate(h_Aptr);
  gt::backend::device_allocator<T*>::deallocate(d_Aptr);
  gt::backend::host_allocator<T>::deallocate(h_A);
  gt::backend::device_allocator<T>::deallocate(d_A);

  gt::backend::host_allocator<gtblas_index_t>::deallocate(h_p);
  gt::backend::device_allocator<gtblas_index_t>::deallocate(d_p);
  gt::backend::host_allocator<int>::deallocate(h_info);
  gt::backend::device_allocator<int>::deallocate(d_info);
}

TEST(getrfs, zgetrs_batch2)
{
  constexpr int N = 3;
  constexpr int NRHS = 2;
  constexpr int S = N * N;
  constexpr int batch_size = 2;
  using T = gt::complex<double>;

  T** h_Aptr = gt::backend::host_allocator<T*>::allocate(batch_size);
  T** d_Aptr = gt::backend::device_allocator<T*>::allocate(batch_size);
  T* h_A = gt::backend::host_allocator<T>::allocate(batch_size * S);
  T* d_A = gt::backend::device_allocator<T>::allocate(batch_size * S);

  T** h_Bptr = gt::backend::host_allocator<T*>::allocate(batch_size);
  T** d_Bptr = gt::backend::device_allocator<T*>::allocate(batch_size);
  T* h_B = gt::backend::host_allocator<T>::allocate(batch_size * N * NRHS);
  T* d_B = gt::backend::device_allocator<T>::allocate(batch_size * N * NRHS);

  auto h_p =
    gt::backend::host_allocator<gtblas_index_t>::allocate(batch_size * N);
  auto d_p =
    gt::backend::device_allocator<gtblas_index_t>::allocate(batch_size * N);

  set_A_LU(h_A);
  set_B_LU(h_A + S);
  h_Aptr[0] = &d_A[0];
  h_Aptr[1] = &d_A[S];
  set_A_piv(h_p);
  set_B_piv(h_p + N);

  // col vector [11; 18; 28]
  h_B[0] = 11;
  h_B[1] = 18;
  h_B[2] = 28;
  // col vector [73; 78; 154]
  h_B[3] = 73;
  h_B[4] = 78;
  h_B[5] = 154;
  // col vector [73; 78; 154]
  h_B[N * NRHS + 0] = T(11, -1);
  h_B[N * NRHS + 1] = T(14, 4);
  h_B[N * NRHS + 2] = T(16, 12);
  // col vector [73-10i; 90-12i; 112 + 42i]
  h_B[N * NRHS + 3] = T(73, -10);
  h_B[N * NRHS + 4] = T(90, -12);
  h_B[N * NRHS + 5] = T(112, 42);

  h_Bptr[0] = &d_B[0];
  h_Bptr[1] = &d_B[N * NRHS];

  gt::backend::device_copy_hd(h_Aptr, d_Aptr, batch_size);
  gt::backend::device_copy_hd(h_A, d_A, batch_size * S);
  gt::backend::device_copy_hd(h_Bptr, d_Bptr, batch_size);
  gt::backend::device_copy_hd(h_B, d_B, batch_size * N * NRHS);
  gt::backend::device_copy_hd(h_B, d_B, batch_size * N * NRHS);
  gt::backend::device_copy_hd(h_p, d_p, batch_size * N);

  gtblas_create();

  gtblas_zgetrs_batched(N, NRHS, (gtblas_complex_double_t**)d_Aptr, N, d_p,
                         (gtblas_complex_double_t**)d_Bptr, N, batch_size);

  gtblas_destroy();

  gt::backend::device_copy_dh(d_B, h_B, batch_size * N * NRHS);

  // solution vector [1; 2; 3]
  expect_complex_eq(h_B[0], 1.0);
  expect_complex_eq(h_B[1], 2.0);
  expect_complex_eq(h_B[2], 3.0);
  // solution vector [-3; 7; 31]
  expect_complex_eq(h_B[3], -3.0);
  expect_complex_eq(h_B[4], 7.0);
  expect_complex_eq(h_B[5], 31.0);
  // solution vector [1; 2; 3]
  expect_complex_eq(h_B[N * NRHS + 0], 1.0);
  expect_complex_eq(h_B[N * NRHS + 1], 2.0);
  expect_complex_eq(h_B[N * NRHS + 2], 3.0);
  // solution vector [-3; 7; 31]
  expect_complex_eq(h_B[N * NRHS + 3], -3.0);
  expect_complex_eq(h_B[N * NRHS + 4], 7.0);
  expect_complex_eq(h_B[N * NRHS + 5], 31.0);

  gt::backend::host_allocator<T*>::deallocate(h_Aptr);
  gt::backend::device_allocator<T*>::deallocate(d_Aptr);
  gt::backend::host_allocator<T>::deallocate(h_A);
  gt::backend::device_allocator<T>::deallocate(d_A);

  gt::backend::host_allocator<T>::deallocate(h_B);
  gt::backend::device_allocator<T>::deallocate(d_B);
  gt::backend::host_allocator<T*>::deallocate(h_Bptr);
  gt::backend::device_allocator<T*>::deallocate(d_Bptr);

  gt::backend::host_allocator<gtblas_index_t>::deallocate(h_p);
  gt::backend::device_allocator<gtblas_index_t>::deallocate(d_p);
}
