#ifndef GTENSOR_SOLVE_H
#define GTENSOR_SOLVE_H

#include "gtensor/gtensor.h"

#include "gt-blas/blas.h"

namespace gt
{

namespace solver
{

template <typename T>
class GpuSolverDense
{
public:
  GpuSolverDense(gt::blas::handle_t& h, int n, int nbatches, int nrhs,
                 T* const* matrix_batches);

  virtual void prepare();

  virtual void solve(T* rhs, T* result);

protected:
  gt::blas::handle_t& h_;
  gt::gtensor_device<T, 3> matrix_data_;
  gt::gtensor_device<T*, 1> matrix_pointers_;
  gt::gtensor_device<gt::blas::index_t, 2> pivot_data_;
  gt::gtensor_device<int, 1> info_;
  gt::gtensor_device<T, 3> rhs_data_;
  gt::gtensor_device<T*, 1> rhs_pointers_;
  int n_;
  int nbatches_;
  int nrhs_;
  bool prepared_;
};

extern template class GpuSolverDense<float>;
extern template class GpuSolverDense<double>;
extern template class GpuSolverDense<gt::complex<float>>;
extern template class GpuSolverDense<gt::complex<double>>;

#ifdef GTENSOR_DEVICE_SYCL

template <typename T>
class GpuSolverDenseSYCL : public GpuSolverDense<T>
{
public:
  GpuSolverDenseSYCL(gt::blas::handle_t& h, int n, int nbatches, int nrhs,
                     T* const* matrix_batches);

  virtual void prepare();

  virtual void solve(T* rhs, T* result);

protected:
  gt::blas::index_t scratch_count_;
  gt::space::device_vector<T> scratch_;
};

extern template class GpuSolverDenseSYCL<float>;
extern template class GpuSolverDenseSYCL<double>;
extern template class GpuSolverDenseSYCL<gt::complex<float>>;
extern template class GpuSolverDenseSYCL<gt::complex<double>>;

#endif

} // namespace solver

} // namespace gt

#endif
