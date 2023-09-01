#ifndef GTENSOR_SOLVE_H
#define GTENSOR_SOLVE_H

#include "gtensor/gtensor.h"
#include "gtensor/sparse.h"

#include "gt-blas/blas.h"
// CUDART_VERSION >= 11310
#ifdef GTENSOR_DEVICE_CUDA
#if CUDART_VERSION >= 12000
#if CUDART_VERSION >= 12010
// New generic API; available since 11.3.1, added support
// for in-place solve in 12.1
#include "gt-solver/backend/cuda-generic.h"
#else
// bsrsm2 API, which can work with csr format by setting
// block size 1. Deprecated in 12.2
// May be faster than generic API in some cuda versions.
// Exists even in 8.0, but not clear it has advantage over
// csrsm2 API for older cuda versions where csrsm2 is still
// available.
#include "gt-solver/backend/cuda-bsrsm2.h"
#endif // version 12010
#else
// legacy API, deprecated since 11.3.1 but still supported until 12
#include "gt-solver/backend/cuda-csrsm2.h"
#endif // CUDA_VERSION

#elif defined(GTENSOR_DEVICE_HIP)
#ifdef GTENSOR_SOLVER_HIP_SPARSE_GENERIC
#include "gt-solver/backend/hip-generic.h"
#else
#include "gt-solver/backend/hip-csrsm2.h"
#endif

#elif defined(GTENSOR_DEVICE_SYCL)
#include "gt-solver/backend/sycl.h"

#elif defined(GTENSOR_DEVICE_HOST)
#include "gt-solver/backend/host.h"
#endif

namespace gt
{

namespace solver
{

template <typename T>
class solver
{
public:
  using value_type = T;

  virtual void solve(T* rhs, T* result) = 0;
  virtual std::size_t get_device_memory_usage() = 0;
};

template <typename Solver>
class staging_solver : solver<typename Solver::value_type>
{
public:
  using value_type = typename Solver::value_type;
  static constexpr bool inplace = Solver::inplace;

  staging_solver(gt::blas::handle_t& h, int n, int nbatches, int nrhs,
                 value_type* const* matrix_batches);

  virtual void solve(value_type* rhs, value_type* result);
  virtual std::size_t get_device_memory_usage();

private:
  int n_;
  int nbatches_;
  int nrhs_;
  Solver solver_;
  gt::gtensor_device<value_type, 3> rhs_stage_;
  value_type* rhs_stage_p_;
  gt::gtensor_device<value_type, 3> result_stage_;
  value_type* result_stage_p_;
};

#ifdef GTENSOR_DEVICE_SYCL

// use contiguous strided dense API for SYCL, it has been better
// optimized

template <typename T>
class solver_dense : public solver<T>
{
public:
  using base_type = solver<T>;
  using typename base_type::value_type;
  static constexpr bool inplace = true;

  solver_dense(gt::blas::handle_t& h, int n, int nbatches, int nrhs,
               T* const* matrix_batches);

  virtual void solve(T* rhs, T* result);
  virtual std::size_t get_device_memory_usage();

protected:
  gt::blas::handle_t& h_;
  int n_;
  int nbatches_;
  int nrhs_;
  gt::gtensor_device<T, 3> matrix_data_;
  gt::gtensor_device<gt::blas::index_t, 2> pivot_data_;
  gt::blas::index_t scratch_count_;
  gt::space::device_vector<T> scratch_;
};

#else // HIP and CUDA

template <typename T>
class solver_dense : public solver<T>
{
public:
  using base_type = solver<T>;
  using typename base_type::value_type;
  static constexpr bool inplace = true;

  solver_dense(gt::blas::handle_t& h, int n, int nbatches, int nrhs,
               T* const* matrix_batches);

  virtual void solve(T* rhs, T* result);
  virtual std::size_t get_device_memory_usage();

protected:
  gt::blas::handle_t& h_;
  int n_;
  int nbatches_;
  int nrhs_;
  gt::gtensor_device<T, 3> matrix_data_;
  gt::gtensor_device<T*, 1> matrix_pointers_;
  gt::gtensor_device<gt::blas::index_t, 2> pivot_data_;
  gt::gtensor_device<int, 1> info_;
  gt::gtensor_device<T*, 1> rhs_pointers_;
  gt::gtensor<T*, 1> h_rhs_pointers_;
};

#endif

template <typename T>
class solver_invert : solver<T>
{
public:
  using base_type = solver<T>;
  using typename base_type::value_type;
  static constexpr bool inplace = false;

  solver_invert(gt::blas::handle_t& h, int n, int nbatches, int nrhs,
                T* const* matrix_batches);

  virtual void solve(T* rhs, T* result);
  virtual std::size_t get_device_memory_usage();

protected:
  gt::blas::handle_t& h_;
  int n_;
  int nbatches_;
  int nrhs_;
  gt::gtensor_device<T, 3> matrix_data_;
  gt::gtensor_device<T*, 1> matrix_pointers_;
  gt::gtensor_device<gt::blas::index_t, 2> pivot_data_;
  gt::gtensor_device<int, 1> info_;
  gt::gtensor_device<T*, 1> rhs_pointers_;
  gt::gtensor<T*, 1> h_rhs_pointers_;
  gt::gtensor_device<T*, 1> result_pointers_;
  gt::gtensor<T*, 1> h_result_pointers_;
};

#ifdef GTENSOR_SOLVER_HAVE_CSR_MATRIX_LU
#define GTENSOR_SOLVER_HAVE_SOLVER_SPARSE

template <typename T>
class solver_sparse : public solver<T>
{
public:
  using base_type = solver<T>;
  using typename base_type::value_type;
  using csr_matrix_type = gt::sparse::csr_matrix<T, gt::space::device>;
  static constexpr bool inplace = csr_matrix_lu<T>::inplace;

  solver_sparse(gt::blas::handle_t& blas_h, int n, int nbatches, int nrhs,
                T* const* matrix_batches);

  virtual void solve(T* rhs, T* result);
  virtual std::size_t get_device_memory_usage();

protected:
  int n_;
  int nbatches_;
  int nrhs_;
  csr_matrix_type csr_mat_;
  csr_matrix_lu<T> csr_mat_lu_;

private:
  static gt::sparse::csr_matrix<T, gt::space::device> lu_factor_batches_to_csr(
    gt::blas::handle_t& h, int n, int nbatches, T* const* matrix_batches);
};

#endif

template <typename T>
class solver_banded : public solver<T>
{
public:
  using base_type = solver<T>;
  using typename base_type::value_type;
  static constexpr bool inplace = true;

  solver_banded(gt::blas::handle_t& h, int n, int nbatches, int nrhs,
                T* const* matrix_batches);

  virtual void solve(T* rhs, T* result);
  virtual std::size_t get_device_memory_usage();

protected:
  gt::blas::handle_t& h_;
  int n_;
  int nbatches_;
  int nrhs_;
  int lbw_;
  int ubw_;
  gt::gtensor_device<T, 3> matrix_data_;
  gt::gtensor_device<T*, 1> matrix_pointers_;
  gt::gtensor_device<gt::blas::index_t, 2> pivot_data_;
  gt::gtensor_device<int, 1> info_;
  gt::gtensor_device<T*, 1> rhs_pointers_;
  gt::gtensor<T*, 1> h_rhs_pointers_;
};

} // namespace solver

} // namespace gt

#endif
