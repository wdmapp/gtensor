#include <iostream>

#include <gtest/gtest.h>

#include "gtensor/gtensor.h"

#include "gt-solver/solver.h"

#include "gtest_predicates.h"
#include "test_debug.h"

template <typename Solver>
void test_full_solve()
{
  using T = typename Solver::value_type;
  constexpr int N = 5;
  constexpr int NRHS = 2;
  constexpr int batch_size = 1;

  gt::gtensor<T*, 1> h_Aptr(gt::shape(batch_size));
  gt::gtensor<T, 3> h_A(gt::shape(N, N, batch_size));

  gt::gtensor_device<T, 3> d_B(gt::shape(N, NRHS, batch_size));
  gt::gtensor<T, 3> h_B(gt::shape(N, NRHS, batch_size));

  gt::gtensor_device<T, 3> d_C(gt::shape(N, NRHS, batch_size));
  gt::gtensor<T, 3> h_C(gt::shape(N, NRHS, batch_size));

  /*
  A = [ 2 -1  0  0  0;
     -1  2 -1  0  0;
      0 -1  2 -1  0;
      0  0 -1  2 -1;
      0  0  0 -1  2]
      */
  for (int i = 0; i < N; i++) {
    h_B(i, 0, 0) = 1.0;
    h_B(i, 1, 0) = -2.0;
    for (int j = 0; j < N; j++) {
      if (i == j) {
        h_A(j, i, 0) = 2.0;
      } else if (std::abs(i - j) == 1) {
        h_A(j, i, 0) = -1.0;
      } else {
        h_A(j, i, 0) = 0.0;
      }
    }
  }
  h_Aptr(0) = gt::raw_pointer_cast(h_A.data());

  gt::blas::handle_t h;

  Solver solver(h, N, batch_size, NRHS, gt::raw_pointer_cast(h_Aptr.data()));

  gt::copy(h_B, d_B);
  solver.solve(gt::raw_pointer_cast(d_B.data()),
               gt::raw_pointer_cast(d_C.data()));
  gt::copy(d_C, h_C);

  gt::gtensor<T, 1> h_C_expected(gt::shape(N));

  h_C_expected(0) = 2.5;
  h_C_expected(1) = 4.0;
  h_C_expected(2) = 4.5;
  h_C_expected(3) = 4.0;
  h_C_expected(4) = 2.5;
  GT_EXPECT_NEAR(h_C.view(gt::all, 0, 0), h_C_expected);

  // second batch should be -2 times first batch
  h_C_expected = T(-2.0) * h_C_expected;
  GT_EXPECT_NEAR(h_C.view(gt::all, 1, 0), h_C_expected);
}

TEST(solver, sfull_dense_solve)
{
  test_full_solve<gt::solver::SolverDense<float>>();
}

TEST(solver, dfull_dense_solve)
{
  test_full_solve<gt::solver::SolverDense<double>>();
}

TEST(solver, cfull_dense_solve)
{
  test_full_solve<gt::solver::SolverDense<gt::complex<float>>>();
}

TEST(solver, zfull_dense_solve)
{
  test_full_solve<gt::solver::SolverDense<gt::complex<double>>>();
}

TEST(solver, sfull_invert_solve)
{
  test_full_solve<gt::solver::SolverInvert<float>>();
}

TEST(solver, dfull_invert_solve)
{
  test_full_solve<gt::solver::SolverInvert<double>>();
}

TEST(solver, cfull_invert_solve)
{
  test_full_solve<gt::solver::SolverInvert<gt::complex<float>>>();
}

TEST(solver, zfull_invert_solve)
{
  test_full_solve<gt::solver::SolverInvert<gt::complex<double>>>();
}

TEST(solver, sfull_sparse_solve)
{
  test_full_solve<gt::solver::SolverSparse<float>>();
}

TEST(solver, dfull_sparse_solve)
{
  test_full_solve<gt::solver::SolverSparse<double>>();
}

// Note: oneMKL sparse API does not support complex yet
#if defined(GTENSOR_DEVICE_CUDA) || defined(GTENSOR_DEVICE_HIP)

TEST(solver, cfull_sparse_solve)
{
  test_full_solve<gt::solver::SolverSparse<gt::complex<float>>>();
}

TEST(solver, zfull_sparse_solve)
{
  test_full_solve<gt::solver::SolverSparse<gt::complex<double>>>();
}

#endif
