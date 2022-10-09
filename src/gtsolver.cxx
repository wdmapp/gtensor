#include "gtensor/gtensor.h"

#include "gt-solver/solver.h"

namespace gt
{

namespace solver
{

template <typename T>
GpuSolverDense<T>::GpuSolverDense(gt::blas::handle_t& h, int n, int nbatches,
                                  int nrhs, T* const* matrix_batches)
  : h_(h),
    n_(n),
    nbatches_(nbatches),
    nrhs_(nrhs),
    matrix_data_(gt::shape(n, n, nbatches)),
    matrix_pointers_(gt::shape(nbatches)),
    pivot_data_(gt::shape(n, nbatches)),
    pivot_pointers_(gt::shape(nbatches)),
    info_(gt::shape(nbatches)),
    rhs_data_(gt::shape(n, nrhs, nbatches)),
    rhs_pointers_(gt::shape(nbatches))
{
  gt::gtensor<T*, 1> h_matrix_pointers(gt::shape(nbatches));
  gt::gtensor<gt::blas::index_t*, 1> h_pivot_pointers(gt::shape(nbatches));
  gt::gtensor<T*, 1> h_rhs_pointers(gt::shape(nbatches));
  auto pdata = matrix_data_.data();
  decltype(pdata) pmatrix;
  T* prhs_data = gt::raw_pointer_cast(rhs_data_.data());
  gt::blas::index_t* ppivot_data = gt::raw_pointer_cast(pivot_data_.data());
  // copy in to contiguous device memory
  for (int i = 0; i < nbatches; i++) {
    pmatrix = pdata + n * n * i;
    gt::copy_n(matrix_batches[i], n * n, pmatrix);
    h_matrix_pointers(i) = gt::raw_pointer_cast(pmatrix);
    h_rhs_pointers(i) = prhs_data + n * nrhs * i;
    h_pivot_pointers(i) = ppivot_data + n * i;
  }
  gt::copy(h_matrix_pointers, matrix_pointers_);
  gt::copy(h_rhs_pointers, rhs_pointers_);
  gt::copy(h_pivot_pointers, pivot_pointers_);
  gt::synchronize();
  gt::blas::getrf_batched<T>(h_, n_,
                             gt::raw_pointer_cast(matrix_pointers_.data()), n_,
                             gt::raw_pointer_cast(pivot_data_.data()),
                             gt::raw_pointer_cast(info_.data()), nbatches_);
  gt::synchronize();
}

template <typename T>
void GpuSolverDense<T>::solve(T* rhs, T* result)
{
  gt::copy_n(gt::device_pointer_cast(rhs), n_ * nrhs_ * nbatches_,
             rhs_data_.data());
  gt::synchronize();
  gt::blas::getrs_batched<T>(
    h_, n_, nrhs_, gt::raw_pointer_cast(matrix_pointers_.data()), n_,
    gt::raw_pointer_cast(pivot_data_.data()),
    gt::raw_pointer_cast(rhs_pointers_.data()), n_, nbatches_);
  gt::synchronize();
  gt::copy_n(rhs_data_.data(), n_ * nrhs_ * nbatches_,
             gt::device_pointer_cast(result));
}

template class GpuSolverDense<float>;
template class GpuSolverDense<double>;
template class GpuSolverDense<gt::complex<float>>;
template class GpuSolverDense<gt::complex<double>>;

} // namespace solver

} // namespace gt
