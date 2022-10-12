#ifndef GTENSOR_SOLVE_H
#define GTENSOR_SOLVE_H

#include "gtensor/gtensor.h"

#include "gt-blas/blas.h"

namespace gt
{

namespace solver
{

template <typename T>
class Solver
{
public:
  using value_type = T;

  virtual void solve(T* rhs, T* result) = 0;

protected:
  virtual void prepare() = 0;
};

#ifdef GTENSOR_DEVICE_SYCL

// use contiguous strided dense API for SYCL, it has been better
// optimized

template <typename T>
class SolverDense : public Solver<T>
{
public:
  using base_type = Solver<T>;
  using typename base_type::value_type;

  SolverDense(gt::blas::handle_t& h, int n, int nbatches, int nrhs,
              T* const* matrix_batches);

  virtual void solve(T* rhs, T* result);

protected:
  virtual void prepare();

  gt::blas::handle_t& h_;
  int n_;
  int nbatches_;
  int nrhs_;
  gt::gtensor_device<T, 3> matrix_data_;
  gt::gtensor_device<gt::blas::index_t, 2> pivot_data_;
  gt::gtensor_device<T, 3> rhs_data_;
  gt::blas::index_t scratch_count_;
  gt::space::device_vector<T> scratch_;
};

#else // HIP and CUDA

template <typename T>
class SolverDense : public Solver<T>
{
public:
  using base_type = Solver<T>;
  using typename base_type::value_type;

  SolverDense(gt::blas::handle_t& h, int n, int nbatches, int nrhs,
              T* const* matrix_batches);

  virtual void solve(T* rhs, T* result);

protected:
  virtual void prepare();

  gt::blas::handle_t& h_;
  int n_;
  int nbatches_;
  int nrhs_;
  gt::gtensor_device<T, 3> matrix_data_;
  gt::gtensor_device<T*, 1> matrix_pointers_;
  gt::gtensor_device<gt::blas::index_t, 2> pivot_data_;
  gt::gtensor_device<int, 1> info_;
  gt::gtensor_device<T, 3> rhs_data_;
  gt::gtensor_device<T*, 1> rhs_pointers_;
};

#endif

template <typename T>
class SolverInvert : Solver<T>
{
public:
  using base_type = Solver<T>;
  using typename base_type::value_type;

  SolverInvert(gt::blas::handle_t& h, int n, int nbatches, int nrhs,
               T* const* matrix_batches);

  virtual void solve(T* rhs, T* result);

protected:
  virtual void prepare();

  gt::blas::handle_t& h_;
  int n_;
  int nbatches_;
  int nrhs_;
  gt::gtensor_device<T, 3> matrix_data_;
  gt::gtensor_device<T*, 1> matrix_pointers_;
  gt::gtensor_device<gt::blas::index_t, 2> pivot_data_;
  gt::gtensor_device<int, 1> info_;
  gt::gtensor_device<T, 3> rhs_data_;
  gt::gtensor_device<T*, 1> rhs_pointers_;
  gt::gtensor_device<T, 3> rhs_input_data_;
  gt::gtensor_device<T*, 1> rhs_input_pointers_;
};

} // namespace solver

} // namespace gt

#endif
