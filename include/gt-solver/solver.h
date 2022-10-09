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

  void solve(T* rhs, T* result);

private:
  gt::blas::handle_t& h_;
  gt::gtensor_device<T, 3> matrix_data_;
  gt::gtensor_device<T*, 1> matrix_pointers_;
  gt::gtensor_device<gt::blas::index_t, 2> pivot_data_;
  gt::gtensor_device<gt::blas::index_t*, 1> pivot_pointers_;
  gt::gtensor_device<int, 1> info_;
  gt::gtensor_device<T, 3> rhs_data_;
  gt::gtensor_device<T*, 1> rhs_pointers_;
  int n_;
  int nbatches_;
  int nrhs_;
};

extern template class GpuSolverDense<float>;
extern template class GpuSolverDense<double>;
extern template class GpuSolverDense<gt::complex<float>>;
extern template class GpuSolverDense<gt::complex<double>>;

} // namespace solver

} // namespace gt

#endif
