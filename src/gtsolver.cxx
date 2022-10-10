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
    info_(gt::shape(nbatches)),
    rhs_data_(gt::shape(n, nrhs, nbatches)),
    rhs_pointers_(gt::shape(nbatches)),
    prepared_(false)
{
  gt::gtensor<T*, 1> h_matrix_pointers(gt::shape(nbatches));
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
  }
  gt::copy(h_matrix_pointers, matrix_pointers_);
  gt::copy(h_rhs_pointers, rhs_pointers_);
}

template <typename T>
void GpuSolverDense<T>::prepare()
{
  gt::blas::getrf_batched<T>(h_, n_,
                             gt::raw_pointer_cast(matrix_pointers_.data()), n_,
                             gt::raw_pointer_cast(pivot_data_.data()),
                             gt::raw_pointer_cast(info_.data()), nbatches_);
  prepared_ = true;
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

#ifdef GTENSOR_DEVICE_SYCL

template <typename T>
GpuSolverDenseSYCL<T>::GpuSolverDenseSYCL(gt::blas::handle_t& h, int n,
                                          int nbatches, int nrhs,
                                          T* const* matrix_batches)
  : GpuSolverDense<T>(h, n, nbatches, nrhs, matrix_batches),
    scratch_count_(gt::blas::getrs_strided_batched_scratchpad_size<T>(
      h, n, n, n, nrhs, nbatches)),
    scratch_(scratch_count_)
{}

template <typename T>
void GpuSolverDenseSYCL<T>::prepare()
{
  auto prep_scratch_count = gt::blas::getrf_strided_batched_scratchpad_size<T>(
    this->h_, this->n_, this->n_, this->nbatches_);
  gt::space::device_vector<T> prep_scratch(prep_scratch_count);

  gt::blas::getrf_strided_batched<T>(
    this->h_, this->n_, gt::raw_pointer_cast(this->matrix_data_.data()),
    this->n_, gt::raw_pointer_cast(this->pivot_data_.data()), this->nbatches_,
    gt::raw_pointer_cast(prep_scratch.data()), prep_scratch_count);
  // Note: synchronize so it's safe to garbage collect scratch
  gt::synchronize();
  this->prepared_ = true;
}

template <typename T>
void GpuSolverDenseSYCL<T>::solve(T* rhs, T* result)
{
  gt::copy_n(gt::device_pointer_cast(rhs),
             this->n_ * this->nrhs_ * this->nbatches_, this->rhs_data_.data());
  gt::synchronize();
  gt::blas::getrs_strided_batched<T>(
    this->h_, this->n_, this->nrhs_,
    gt::raw_pointer_cast(this->matrix_data_.data()), this->n_,
    gt::raw_pointer_cast(this->pivot_data_.data()),
    gt::raw_pointer_cast(this->rhs_data_.data()), this->n_, this->nbatches_,
    gt::raw_pointer_cast(this->scratch_.data()), this->scratch_count_);
  gt::synchronize();
  gt::copy_n(this->rhs_data_.data(), this->n_ * this->nrhs_ * this->nbatches_,
             gt::device_pointer_cast(result));
}

template class GpuSolverDenseSYCL<float>;
template class GpuSolverDenseSYCL<double>;
template class GpuSolverDenseSYCL<gt::complex<float>>;
template class GpuSolverDenseSYCL<gt::complex<double>>;

#endif

} // namespace solver

} // namespace gt
