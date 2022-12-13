#include <iostream>

#include <gtest/gtest.h>

#include "gtensor/gtensor.h"
#include "gtensor/sparse.h"

#include "gtest_predicates.h"
#include "test_debug.h"

namespace detail
{

template <typename T>
GT_INLINE T norm(gt::complex<T> a)
{
  return gt::norm(a);
}

GT_INLINE double norm(double a)
{
  return gt::abs(a);
}

GT_INLINE float norm(float a)
{
  return gt::abs(a);
}

} // namespace detail

template <typename T, typename S>
void test_csr_matrix_batched()
{
  constexpr int N = 5;
  constexpr int NBATCHES = 100;
  gt::gtensor<T, 3> h_A(gt::shape(N, N, NBATCHES));
  gt::gtensor<T, 3, S> d_A(h_A.shape());

  /*
  A_b = [ (b, -b) -1  0  0  0;
         -1  (b, -b) -1  0  0;
          0 -1  (b, -b) -1  0;
          0  0 -1  (b, -b) -1;
          0  0  0 -1  (b, -b) ]
      */
  for (int b = 0; b < NBATCHES; b++) {
    for (int j = 0; j < N; j++) {
      for (int i = 0; i < N; i++) {
        if (i == j) {
          h_A(i, j, b) = T(b, -b);
        } else if (std::abs(i - j) == 1) {
          h_A(i, j, b) = -1.0;
        } else {
          h_A(i, j, b) = 0.0;
        }
      }
    }
  }

  gt::copy(h_A, d_A);
  gt::synchronize();

  auto d_Acsr = gt::sparse::csr_matrix<T, S>::join_matrix_batches(d_A);

  gt::gtensor<double, 1, S> d_err{gt::shape(1)};
  gt::gtensor<double, 1> h_err{gt::shape(1)};
  auto k_err = d_err.to_kernel();
  auto k_Acsr = d_Acsr.to_kernel();

  gt::launch<1, S>(
    gt::shape(NBATCHES), GT_LAMBDA(int b) {
      k_err(0) = 0.0;
      for (int b = 0; b < NBATCHES; b++) {
        for (int j = b * N; j < (b + 1) * N; j++) {
          for (int i = b * N; i < (b + 1) * N; i++) {
            if (i == j) {
              k_err(0) += gt::norm(k_Acsr(i, j) - T(b, -b));
            } else if (std::abs(i - j) == 1) {
              k_err(0) += gt::norm(k_Acsr(i, j) - T(-1, 0));
            } else {
              k_err(0) += gt::norm(k_Acsr(i, j));
            }
          }
        }
      }
    });

  gt::copy(d_err, h_err);

  EXPECT_EQ(h_err(0), 0);
}

TEST(sparse, csr_matrix_batched_host_z)
{
  test_csr_matrix_batched<gt::complex<double>, gt::space::host>();
}

TEST(sparse, csr_matrix_batched_host_c)
{
  test_csr_matrix_batched<gt::complex<float>, gt::space::host>();
}

#ifdef GTENSOR_HAVE_DEVICE

TEST(sparse, csr_matrix_batched_device_z)
{
  test_csr_matrix_batched<gt::complex<double>, gt::space::device>();
}

TEST(sparse, csr_matrix_batched_device_c)
{
  test_csr_matrix_batched<gt::complex<float>, gt::space::device>();
}

#endif

template <typename T, typename S>
void test_csr_matrix()
{
  constexpr int N = 5;
  gt::gtensor<T, 2> h_A(gt::shape(N, N));
  gt::gtensor<T, 2, S> d_A(h_A.shape());

  /*
  A_b = [ 2 -1  0  0  0;
         -1  2 -1  0  0;
          0 -1  2 -1  0;
          0  0 -1  2 -1;
          0  0  0 -1  2 ]
      */
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < N; i++) {
      if (i == j) {
        h_A(i, j) = 2.0;
      } else if (std::abs(i - j) == 1) {
        h_A(i, j) = -1.0;
      } else {
        h_A(i, j) = 0.0;
      }
    }
  }

  gt::copy(h_A, d_A);
  gt::synchronize();

  gt::sparse::csr_matrix<T, S> d_Acsr{d_A};

  gt::gtensor<double, 1, S> d_err{gt::shape(1)};
  gt::gtensor<double, 1> h_err{gt::shape(1)};
  auto k_err = d_err.to_kernel();
  auto k_Acsr = d_Acsr.to_kernel();

  gt::launch<1, S>(
    gt::shape(1), GT_LAMBDA(int b) {
      k_err(0) = 0.0;
      for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
          if (i == j) {
            k_err(0) += detail::norm(k_Acsr(i, j) - T(2));
          } else if (std::abs(i - j) == 1) {
            k_err(0) += detail::norm(k_Acsr(i, j) - T(-1));
          } else {
            k_err(0) += detail::norm(k_Acsr(i, j));
          }
        }
      }
    });

  gt::copy(d_err, h_err);

  EXPECT_EQ(h_err(0), 0);
}

TEST(sparse, csr_matrix_host_s)
{
  test_csr_matrix<float, gt::space::host>();
}

TEST(sparse, csr_matrix_host_c)
{
  test_csr_matrix<gt::complex<float>, gt::space::host>();
}

#ifdef GTENSOR_HAVE_DEVICE

TEST(sparse, csr_matrix_device_s)
{
  test_csr_matrix<float, gt::space::device>();
}

TEST(sparse, csr_matrix_device_c)
{
  test_csr_matrix<gt::complex<float>, gt::space::device>();
}

#endif

TEST(sparse, csr_matrix_function)
{
  gt::gtensor<double, 2> a{
    {2, -1, 0, 0}, {-1, 2, -1, 0}, {0, -1, 2, -1}, {0, 0, -1, 2}};
  gt::sparse::csr_matrix<double, gt::space::host> s_a(a);

  auto b = s_a * 2;
  EXPECT_EQ(b.shape(), a.shape());

  GT_DEBUG_TYPE(b);

  auto beval = gt::eval(b);

  GT_DEBUG_TYPE(beval);

  GT_EXPECT_EQ(beval, (2 * a));
}

TEST(sparse, csr_matrix_view)
{
  gt::gtensor<double, 2> a{
    {2, -1, 0, 0}, {-1, 2, -1, 0}, {0, -1, 2, -1}, {0, 0, -1, 2}};
  gt::sparse::csr_matrix<double, gt::space::host> s_a(a);

  auto s_aview = gt::view(s_a, gt::all, gt::slice(0, 2));

  GT_DEBUG_TYPE(s_aview);
  GT_EXPECT_EQ(s_aview, a.view(gt::all, gt::slice(0, 2)));

  auto s_aview_eval = gt::eval(s_aview);
  GT_DEBUG_TYPE(s_aview_eval);
  EXPECT_EQ(s_aview_eval.shape(), gt::shape(4, 2));
}
