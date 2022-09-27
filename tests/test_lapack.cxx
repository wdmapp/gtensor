#include <gtest/gtest.h>

#include "gtensor/gtensor.h"

#include "gt-blas/blas.h"

#include "test_helpers.h"

using namespace std::complex_literals;

// matlab/octave:
//  L,U,p = lu(A)

// clang-format off
// A0 = [1 2 2; 4 4 2; 4 6 4];
gt::gtensor<double, 2> A0{{1, 4, 4},
                          {2, 4, 6},
                          {2, 2, 4}};

gt::gtensor<double, 2> A0_LU{{4.0, 1.0, 0.25},
                             {4.0, 2.0, 0.5 },
                             {2.0, 2.0, 0.5 }};

gt::gtensor<int, 1> A0_piv{2, 3, 3};

// A0_nopiv = [1 0 0; 1 2 0; 0 2 3];
gt::gtensor<double, 2> A0_nopiv{{1, 0, 0},
                                {1, 2, 0},
                                {0, 2, 3}};

// A1 = [1+i 2-i 2; 4i 4 2; 4 6i 4];
gt::gtensor<gt::complex<double>, 2> A1{{1. + 1.i, 4.i, 4. },
                                       {2. - 1.i,  4., 6.i},
                                       {2.      ,  2., 4. }};
// A1_LU
gt::gtensor<gt::complex<double>, 2> A1_LU{{4.i,    - 1.i, 0.25 - 0.25i},
                                          {4. ,     10.i,      - 0.10i},
                                          {2. , 4. + 2.i, 1.30 + 0.90i}};

gt::gtensor<int, 1> A1_piv{2, 3, 3};

// A1_nopiv
gt::gtensor<gt::complex<double>, 2> A1_nopiv{{1., 0., 0.},
                                             {1., 2., 0.},
                                             {0., 2., 3.}};
// clang-format on

template <typename T>
void test_getrf_batch_real()
{
  constexpr int N = 3;
  constexpr int batch_size = 1;

  gt::gtensor<T*, 1> h_Aptr(batch_size);
  gt::gtensor_device<T*, 1> d_Aptr(batch_size);
  gt::gtensor<T, 3> h_A(gt::shape(N, N, batch_size));
  gt::gtensor_device<T, 3> d_A(gt::shape(N, N, batch_size));
  gt::gtensor<gt::blas::index_t, 2> h_p(gt::shape(N, batch_size));
  gt::gtensor_device<gt::blas::index_t, 2> d_p(gt::shape(N, batch_size));
  gt::gtensor<int, 1> h_info(batch_size);
  gt::gtensor_device<int, 1> d_info(batch_size);

  auto h_A0 = h_A.view(gt::all, gt::all, 0);

  h_A0 = A0;
  h_Aptr(0) = gt::raw_pointer_cast(d_A.data());

  gt::copy(h_A, d_A);
  gt::copy(h_Aptr, d_Aptr);

  gt::blas::handle_t h;

  gt::blas::getrf_batched(h, N, gt::raw_pointer_cast(d_Aptr.data()), N,
                          gt::raw_pointer_cast(d_p.data()),
                          gt::raw_pointer_cast(d_info.data()), batch_size);

  gt::copy(d_A, h_A);
  gt::copy(d_p, h_p);
  gt::copy(d_info, h_info);

  // first column factored
  EXPECT_EQ(h_A0(0, 0), 4.0);
  EXPECT_EQ(h_A0(1, 0), 1.0);
  EXPECT_EQ(h_A0(2, 0), 0.25);
  // second column factored
  EXPECT_EQ(h_A0(0, 1), 4.0);
  EXPECT_EQ(h_A0(1, 1), 2.0);
  EXPECT_EQ(h_A0(2, 1), 0.5);
  // third column factored
  EXPECT_EQ(h_A0(0, 2), 2.0);
  EXPECT_EQ(h_A0(1, 2), 2.0);
  EXPECT_EQ(h_A0(2, 2), 0.5);

  // Math notation is 2, 3, 1, but BLAS uses procedural notation, i.e.
  // on the thirst step no swap is done so one-based index of third row
  // is still 3 (no swapping).
  EXPECT_EQ(h_p(0, 0), 2);
  EXPECT_EQ(h_p(1, 0), 3);
  EXPECT_EQ(h_p(2, 0), 3);

  for (int b = 0; b < batch_size; b++) {
    // A_i factored successfully
    EXPECT_EQ(h_info(b), 0);
  }
}

TEST(lapack, sgetrf_batch)
{
  test_getrf_batch_real<float>();
}

TEST(lapack, dgetrf_batch)
{
  test_getrf_batch_real<double>();
}

template <typename T>
void test_getrs_batch_real()
{
  constexpr int N = 3;
  constexpr int NRHS = 2;
  constexpr int batch_size = 1;

  gt::gtensor<T*, 1> h_Aptr(batch_size);
  gt::gtensor_device<T*, 1> d_Aptr(batch_size);
  gt::gtensor<T, 3> h_A(gt::shape(N, N, batch_size));
  gt::gtensor_device<T, 3> d_A(gt::shape(N, N, batch_size));

  gt::gtensor<T*, 1> h_Bptr(batch_size);
  gt::gtensor_device<T*, 1> d_Bptr(batch_size);
  gt::gtensor<T, 3> h_B(gt::shape(N, NRHS, batch_size));
  gt::gtensor_device<T, 3> d_B(gt::shape(N, NRHS, batch_size));

  gt::gtensor<gt::blas::index_t, 2> h_p(gt::shape(N, batch_size));
  gt::gtensor_device<gt::blas::index_t, 2> d_p(gt::shape(N, batch_size));

  // set up first (and only) batch
  h_A.view(gt::all, gt::all, 0) = A0_LU;
  h_p.view(gt::all, 0) = A0_piv;

  // set up two col vectors to solve in first (and only) batch
  // first RHS, col vector [11; 18; 28]
  h_B(0, 0, 0) = 11;
  h_B(1, 0, 0) = 18;
  h_B(2, 0, 0) = 28;
  // second RHS, col vector [73; 78; 154]
  h_B(0, 1, 0) = 73;
  h_B(1, 1, 0) = 78;
  h_B(2, 1, 0) = 154;

  h_Aptr(0) = gt::raw_pointer_cast(d_A.data());
  h_Bptr(0) = gt::raw_pointer_cast(d_B.data());

  gt::copy(h_Aptr, d_Aptr);
  gt::copy(h_A, d_A);
  gt::copy(h_Bptr, d_Bptr);
  gt::copy(h_B, d_B);
  gt::copy(h_p, d_p);

  gt::blas::handle_t h;

  gt::blas::getrs_batched(h, N, NRHS, gt::raw_pointer_cast(d_Aptr.data()), N,
                          gt::raw_pointer_cast(d_p.data()),
                          gt::raw_pointer_cast(d_Bptr.data()), N, batch_size);

  gt::copy(d_B, h_B);

  // solution vector [1; 2; 3]
  EXPECT_EQ(h_B(0, 0, 0), 1.0);
  EXPECT_EQ(h_B(1, 0, 0), 2.0);
  EXPECT_EQ(h_B(2, 0, 0), 3.0);
  // solution vector [-3; 7; 31]
  EXPECT_EQ(h_B(0, 1, 0), -3.0);
  EXPECT_EQ(h_B(1, 1, 0), 7.0);
  EXPECT_EQ(h_B(2, 1, 0), 31.0);
}

TEST(lapack, sgetrs_batch)
{
  test_getrs_batch_real<float>();
}

TEST(lapack, dgetrs_batch)
{
  test_getrs_batch_real<double>();
}

template <typename R>
void test_getrf_batch_complex()
{
  constexpr int N = 3;
  constexpr int batch_size = 2;
  using T = gt::complex<R>;

  gt::gtensor<T*, 1> h_Aptr(batch_size);
  gt::gtensor_device<T*, 1> d_Aptr(batch_size);
  gt::gtensor<T, 3> h_A(gt::shape(N, N, batch_size));
  gt::gtensor_device<T, 3> d_A(gt::shape(N, N, batch_size));
  gt::gtensor<gt::blas::index_t, 2> h_p(gt::shape(N, batch_size));
  gt::gtensor_device<gt::blas::index_t, 2> d_p(gt::shape(N, batch_size));
  gt::gtensor<int, 1> h_info(batch_size);
  gt::gtensor_device<int, 1> d_info(batch_size);

  // setup first batch matrix input
  h_A.view(gt::all, gt::all, 0) = A0;

  // setup second batch matrix input
  h_A.view(gt::all, gt::all, 1) = A1;
  // TODO: better notation for this, i.e. the ability to get a pointer from a
  // view if it is wrapping a gcontainer or gtensor_span?
  h_Aptr(0) = gt::raw_pointer_cast(d_A.data());
  h_Aptr(1) = h_Aptr(0) + N * N;

  gt::copy(h_A, d_A);
  gt::copy(h_Aptr, d_Aptr);

  gt::blas::handle_t h;

  gt::blas::getrf_batched(h, N, gt::raw_pointer_cast(d_Aptr.data()), N,
                          gt::raw_pointer_cast(d_p.data()),
                          gt::raw_pointer_cast(d_info.data()), batch_size);

  gt::copy(d_A, h_A);
  gt::copy(d_p, h_p);
  gt::copy(d_info, h_info);

  // first batch matrix result
  // first column factored
  expect_complex_near(h_A(0, 0, 0), 4.0);
  expect_complex_near(h_A(1, 0, 0), 1.0);
  expect_complex_near(h_A(2, 0, 0), 0.25);
  // second column factored
  expect_complex_near(h_A(0, 1, 0), 4.0);
  expect_complex_near(h_A(1, 1, 0), 2.0);
  expect_complex_near(h_A(2, 1, 0), 0.5);
  // third column factored
  expect_complex_near(h_A(0, 2, 0), 2.0);
  expect_complex_near(h_A(1, 2, 0), 2.0);
  expect_complex_near(h_A(2, 2, 0), 0.5);

  // Math notation is 2, 3, 1, but BLAS uses procedural notation, i.e.
  // on the thirst step no swap is done so one-based index of third row
  // is still 3 (no swapping).
  EXPECT_EQ(h_p(0, 0), 2);
  EXPECT_EQ(h_p(1, 0), 3);
  EXPECT_EQ(h_p(2, 0), 3);

  // second batch matrix result
  // first column factored
  expect_complex_near(h_A(0, 0, 1), T(0, 4));
  expect_complex_near(h_A(1, 0, 1), T(0, -1));
  expect_complex_near(h_A(2, 0, 1), T(0.25, -0.25));
  // second column factored
  expect_complex_near(h_A(0, 1, 1), T(4, 0));
  expect_complex_near(h_A(1, 1, 1), T(0, 10));
  expect_complex_near(h_A(2, 1, 1), T(0, -0.1));
  // third column factored
  expect_complex_near(h_A(0, 2, 1), T(2, 0));
  expect_complex_near(h_A(1, 2, 1), T(4, 2));
  expect_complex_near(h_A(2, 2, 1), T(1.3, 0.9));

  // Math notation is 2, 3, 1, but BLAS uses procedural notation, i.e.
  // on the thirst step no swap is done so one-based index of third row
  // is still 3 (no swapping).
  EXPECT_EQ(h_p(0, 1), 2);
  EXPECT_EQ(h_p(1, 1), 3);
  EXPECT_EQ(h_p(2, 1), 3);

  for (int b = 0; b < batch_size; b++) {
    // A_i factored successfully
    EXPECT_EQ(h_info(b), 0);
  }
}

TEST(lapack, cgetrf_batch)
{
  test_getrf_batch_complex<float>();
}

TEST(lapack, zgetrf_batch)
{
  test_getrf_batch_complex<double>();
}

template <typename R>
void test_getrf_npvt_batch_complex()
{
  constexpr int N = 3;
  constexpr int batch_size = 2;
  using T = gt::complex<R>;

  gt::gtensor<T*, 1> h_Aptr(batch_size);
  gt::gtensor_device<T*, 1> d_Aptr(batch_size);
  gt::gtensor<T, 3> h_A(gt::shape(N, N, batch_size));
  gt::gtensor_device<T, 3> d_A(gt::shape(N, N, batch_size));
  gt::gtensor<int, 1> h_info(batch_size);
  gt::gtensor_device<int, 1> d_info(batch_size);

  // setup first batch matrix input
  h_A.view(gt::all, gt::all, 0) = A0_nopiv;
  h_Aptr(0) = gt::raw_pointer_cast(d_A.data());

  // setup second batch matrix input
  set_A1_complex_nopiv(h_A.view(gt::all, gt::all, 1));
  // TODO: better notation for this, i.e. the ability to get a pointer from a
  // view if it is wrapping a gcontainer or gtensor_span?
  h_Aptr(1) = h_Aptr(0) + N * N;

  gt::copy(h_A, d_A);
  gt::copy(h_Aptr, d_Aptr);

  gt::blas::handle_t h;

  gt::blas::getrf_npvt_batched(h, N, gt::raw_pointer_cast(d_Aptr.data()), N,
                               gt::raw_pointer_cast(d_info.data()), batch_size);

  gt::copy(d_A, h_A);
  gt::copy(d_info, h_info);

  // first batch matrix result
  // first column factored
  expect_complex_near(h_A(0, 0, 0), 1.0);
  expect_complex_near(h_A(1, 0, 0), 0.0);
  expect_complex_near(h_A(2, 0, 0), 0.0);
  // second column factored
  expect_complex_near(h_A(0, 1, 0), 1.0);
  expect_complex_near(h_A(1, 1, 0), 2.0);
  expect_complex_near(h_A(2, 1, 0), 0.0);
  // third column factored
  expect_complex_near(h_A(0, 2, 0), 0.0);
  expect_complex_near(h_A(1, 2, 0), 2.0);
  expect_complex_near(h_A(2, 2, 0), 3.0);

  // second batch matrix result
  // first column factored
  expect_complex_near(h_A(0, 0, 1), T(1, 0));
  expect_complex_near(h_A(1, 0, 1), T(0, 0));
  expect_complex_near(h_A(2, 0, 1), T(0, 0));
  // second column factored
  expect_complex_near(h_A(0, 1, 1), T(1, 0));
  expect_complex_near(h_A(1, 1, 1), T(2, 0));
  expect_complex_near(h_A(2, 1, 1), T(0, 0));
  // third column factored
  expect_complex_near(h_A(0, 2, 1), T(0, 0));
  expect_complex_near(h_A(1, 2, 1), T(2, 0));
  expect_complex_near(h_A(2, 2, 1), T(3, 0));

  for (int b = 0; b < batch_size; b++) {
    // A_i factored successfully
    EXPECT_EQ(h_info(b), 0);
  }
}

TEST(lapack, cgetrf_npvt_batch)
{
  test_getrf_npvt_batch_complex<float>();
}

TEST(lapack, zgetrf_npvt_batch)
{
  test_getrf_npvt_batch_complex<double>();
}

namespace test
{
// little hack to make tests parameterizable on managed vs device memory

template <typename T, gt::size_type N, typename S = gt::space::device>
struct gthelper
{
  using gtensor = gt::gtensor<T, N, S>;
};

template <typename T, gt::size_type N>
struct gthelper<T, N, gt::space::managed>
{
  using gtensor = gt::gtensor_container<gt::space::managed_vector<T>, N>;
};

template <typename T, gt::size_type N, typename S = gt::space::device>
using gtensor2 = typename gthelper<T, N, S>::gtensor;
} // namespace test

template <typename R, typename S = gt::space::device>
void test_getrs_batch_complex()
{
  constexpr int N = 3;
  constexpr int NRHS = 2;
  constexpr int batch_size = 2;
  using T = gt::complex<R>;

  gt::gtensor<T*, 1> h_Aptr(batch_size);
  test::gtensor2<T*, 1, S> d_Aptr(batch_size);
  gt::gtensor<T, 3> h_A(gt::shape(N, N, batch_size));
  test::gtensor2<T, 3, S> d_A(gt::shape(N, N, batch_size));

  gt::gtensor<T*, 1> h_Bptr(batch_size);
  test::gtensor2<T*, 1, S> d_Bptr(batch_size);
  gt::gtensor<T, 3> h_B(gt::shape(N, NRHS, batch_size));
  test::gtensor2<T, 3, S> d_B(gt::shape(N, NRHS, batch_size));

  gt::gtensor<gt::blas::index_t, 2> h_p(gt::shape(N, batch_size));
  test::gtensor2<gt::blas::index_t, 2, S> d_p(gt::shape(N, batch_size));

  // setup input for first batch
  h_A.view(gt::all, gt::all, 0) = A0_LU;
  h_p.view(gt::all, 0) = A0_piv;

  // setup input for second batch
  h_A.view(gt::all, gt::all, 1) = A1_LU;
  h_p.view(gt::all, 1) = A1_piv;

  h_Aptr(0) = gt::raw_pointer_cast(d_A.data());
  h_Aptr[1] = h_Aptr(0) + N * N;

  // first batch, first rhs col vector   (11; 18; 28)
  h_B(0, 0, 0) = 11;
  h_B(1, 0, 0) = 18;
  h_B(2, 0, 0) = 28;
  // first batch, second rhs col vector  (73; 78; 154)
  h_B(0, 1, 0) = 73;
  h_B(1, 1, 0) = 78;
  h_B(2, 1, 0) = 154;
  // second batch, first rhs col vector  (73; 78; 154)
  h_B(0, 0, 1) = T(11, -1);
  h_B(1, 0, 1) = T(14, 4);
  h_B(2, 0, 1) = T(16, 12);
  // second batch, second rhs col vector (73-10i; 90-12i; 112 + 42i)
  h_B(0, 1, 1) = T(73, -10);
  h_B(1, 1, 1) = T(90, -12);
  h_B(2, 1, 1) = T(112, 42);

  h_Bptr(0) = gt::raw_pointer_cast(d_B.data());
  h_Bptr(1) = h_Bptr(0) + N * NRHS;

  gt::copy(h_Aptr, d_Aptr);
  gt::copy(h_A, d_A);
  gt::copy(h_Bptr, d_Bptr);
  gt::copy(h_B, d_B);
  gt::copy(h_B, d_B);
  gt::copy(h_p, d_p);

  gt::blas::handle_t h;

  gt::blas::getrs_batched(h, N, NRHS, gt::raw_pointer_cast(d_Aptr.data()), N,
                          gt::raw_pointer_cast(d_p.data()),
                          gt::raw_pointer_cast(d_Bptr.data()), N, batch_size);

  gt::copy(d_B, h_B);

  // first batch, first solution vector [1; 2; 3]
  expect_complex_near(h_B(0, 0, 0), 1.0);
  expect_complex_near(h_B(1, 0, 0), 2.0);
  expect_complex_near(h_B(2, 0, 0), 3.0);
  // first batch, second solution vector [-3; 7; 31]
  expect_complex_near(h_B(0, 1, 0), -3.0);
  expect_complex_near(h_B(1, 1, 0), 7.0);
  expect_complex_near(h_B(2, 1, 0), 31.0);
  // second batch, first solution vector [1; 2; 3]
  expect_complex_near(h_B(0, 0, 1), 1.0);
  expect_complex_near(h_B(1, 0, 1), 2.0);
  expect_complex_near(h_B(2, 0, 1), 3.0);
  // second batch, second solution vector [-3; 7; 31]
  expect_complex_near(h_B(0, 1, 1), -3.0);
  expect_complex_near(h_B(1, 1, 1), 7.0);
  expect_complex_near(h_B(2, 1, 1), 31.0);
}

TEST(lapack, cgetrs_batch)
{
  test_getrs_batch_complex<float>();
}

TEST(lapack, zgetrs_batch)
{
  test_getrs_batch_complex<double>();
}

TEST(lapack, cgetrs_batch_managed)
{
  test_getrs_batch_complex<float, gt::space::managed>();
}

TEST(lapack, zgetrs_batch_managed)
{
  test_getrs_batch_complex<double, gt::space::managed>();
}

template <typename R, typename S = gt::space::device>
void test_getri_batch_complex()
{
  constexpr int N = 3;
  constexpr int batch_size = 2;
  using T = gt::complex<R>;

  gt::gtensor<T*, 1> h_Aptr(batch_size);
  test::gtensor2<T*, 1, S> d_Aptr(batch_size);
  gt::gtensor<T, 3> h_A(gt::shape(N, N, batch_size));
  test::gtensor2<T, 3, S> d_A(gt::shape(N, N, batch_size));

  gt::gtensor<T*, 1> h_Cptr(batch_size);
  test::gtensor2<T*, 1, S> d_Cptr(batch_size);
  gt::gtensor<T, 3> h_C(gt::shape(N, N, batch_size));
  test::gtensor2<T, 3, S> d_C(gt::shape(N, N, batch_size));

  gt::gtensor<gt::blas::index_t, 2> h_p(gt::shape(N, batch_size));
  test::gtensor2<gt::blas::index_t, 2, S> d_p(gt::shape(N, batch_size));

  gt::gtensor<int, 1> h_info(batch_size);
  gt::gtensor_device<int, 1> d_info(batch_size);

  // setup input for first batch
  h_A.view(gt::all, gt::all, 0) = A0_LU;
  h_p.view(gt::all, 0) = A0_piv;

  // setup input for second batch
  h_A.view(gt::all, gt::all, 1) = A1_LU;
  h_p.view(gt::all, 1) = A1_piv;

  h_Aptr(0) = gt::raw_pointer_cast(d_A.data());
  h_Aptr[1] = h_Aptr(0) + N * N;

  h_Cptr(0) = gt::raw_pointer_cast(d_C.data());
  h_Cptr(1) = h_Cptr(0) + N * N;

  gt::copy(h_Aptr, d_Aptr);
  gt::copy(h_A, d_A);
  gt::copy(h_Cptr, d_Cptr);
  gt::copy(h_C, d_C);
  gt::copy(h_C, d_C);
  gt::copy(h_p, d_p);

  gt::blas::handle_t h;

  gt::blas::getri_batched(h, N, gt::raw_pointer_cast(d_Aptr.data()), N,
                          gt::raw_pointer_cast(d_p.data()),
                          gt::raw_pointer_cast(d_Cptr.data()), N,
                          gt::raw_pointer_cast(d_info.data()), batch_size);

  gt::copy(d_C, h_C);

  // first batch, first solution col [1; -2; 2]
  expect_complex_near(h_C(0, 0, 0), 1.0);
  expect_complex_near(h_C(1, 0, 0), -2.0);
  expect_complex_near(h_C(2, 0, 0), 2.0);
  // first batch, second solution col [1; -1; 0.5]
  expect_complex_near(h_C(0, 1, 0), 1.0);
  expect_complex_near(h_C(1, 1, 0), -1.0);
  expect_complex_near(h_C(2, 1, 0), 0.5);
  // first batch, third solution col [-1; 1.5; -1]
  expect_complex_near(h_C(0, 2, 0), -1.0);
  expect_complex_near(h_C(1, 2, 0), 1.5);
  expect_complex_near(h_C(2, 2, 0), -1.0);
  // second batch, first solution col
  expect_complex_near(h_C(0, 0, 1), T(-0.1, 0.3));
  expect_complex_near(h_C(1, 0, 1), T(0.04, 0.28));
  expect_complex_near(h_C(2, 0, 1), T(0.52, -0.36));
  // second batch, second solution col
  expect_complex_near(h_C(0, 1, 1), T(-0.04, -0.28));
  expect_complex_near(h_C(1, 1, 1), T(0.016, -0.088));
  expect_complex_near(h_C(2, 1, 1), T(-0.092, 0.256));
  // second batch, third solution col
  expect_complex_near(h_C(0, 2, 1), T(0.07, -0.01));
  expect_complex_near(h_C(1, 2, 1), T(-0.028, -0.096));
  expect_complex_near(h_C(2, 2, 1), T(0.036, 0.052));
}

TEST(lapack, cgetri_batch)
{
  test_getri_batch_complex<float>();
}

TEST(lapack, zgetri_batch)
{
  test_getri_batch_complex<double>();
}

TEST(lapack, cgetri_batch_managed)
{
  test_getri_batch_complex<float, gt::space::managed>();
}

TEST(lapack, zgetri_batch_managed)
{
  test_getri_batch_complex<double, gt::space::managed>();
}
