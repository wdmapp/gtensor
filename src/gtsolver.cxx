#include "gtensor/gtensor.h"

#include "gt-solver/solver.h"

namespace gt
{

namespace solver
{

namespace detail
{

template <typename PtrArray, typename DataArray>
void init_device_pointer_array(PtrArray& d_ptr_array, DataArray& d_data_array)
{
  static_assert(gt::expr_dimension<PtrArray>() == 1, "PtrArray must be dim 1");
  static_assert(gt::expr_dimension<DataArray>() == 3,
                "DataArray must be dim 3");
  using P = typename PtrArray::value_type;
  gt::gtensor<P, 1> h_ptr_array(d_ptr_array.shape());
  int nbatches = d_data_array.shape(2);
  for (int i = 0; i < nbatches; i++) {
    h_ptr_array(i) = &(d_data_array(0, 0, i));
  }
  gt::copy(h_ptr_array, d_ptr_array);
}

template <typename T, typename DataArray>
void copy_batch_data(T* const* in_matrix_batches, DataArray& out_data)
{
  static_assert(gt::expr_dimension<DataArray>() == 3,
                "DataArray must be dim 3");
  int nbatches = out_data.shape(2);
  int stride = out_data.shape(0) * out_data.shape(1);
  // copy from pointer per batch to appropriate offset in contiguous device
  // memory
  for (int i = 0; i < nbatches; i++) {
    gt::copy_n(in_matrix_batches[i], stride, out_data.data() + stride * i);
  }
}

} // namespace detail

template <typename T>
SolverDense<T>::SolverDense(gt::blas::handle_t& h, int n, int nbatches,
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
    rhs_pointers_(gt::shape(nbatches))
{
  // copy non-contiguous host memory to contiguous device memory
  detail::copy_batch_data(matrix_batches, matrix_data_);
  detail::init_device_pointer_array(matrix_pointers_, matrix_data_);
  detail::init_device_pointer_array(rhs_pointers_, rhs_data_);
  prepare();
}

template <typename T>
void SolverDense<T>::prepare()
{
  gt::blas::getrf_batched<T>(h_, n_,
                             gt::raw_pointer_cast(matrix_pointers_.data()), n_,
                             gt::raw_pointer_cast(pivot_data_.data()),
                             gt::raw_pointer_cast(info_.data()), nbatches_);
}

template <typename T>
void SolverDense<T>::solve(T* rhs, T* result)
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

template class SolverDense<float>;
template class SolverDense<double>;
template class SolverDense<gt::complex<float>>;
template class SolverDense<gt::complex<double>>;

template <typename T>
SolverInvert<T>::SolverInvert(gt::blas::handle_t& h, int n, int nbatches,
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
    rhs_input_data_(gt::shape(n, nrhs, nbatches)),
    rhs_input_pointers_(gt::shape(nbatches))
{
  detail::copy_batch_data(matrix_batches, matrix_data_);
  detail::init_device_pointer_array(matrix_pointers_, matrix_data_);
  detail::init_device_pointer_array(rhs_pointers_, rhs_data_);
  detail::init_device_pointer_array(rhs_input_pointers_, rhs_input_data_);
  prepare();
}

template <typename T>
void SolverInvert<T>::prepare()
{
  // factor into matrix_data_
  gt::blas::getrf_batched<T>(h_, n_,
                             gt::raw_pointer_cast(matrix_pointers_.data()), n_,
                             gt::raw_pointer_cast(pivot_data_.data()),
                             gt::raw_pointer_cast(info_.data()), nbatches_);

  // getri is not in place, copy input data to temporary, so inverted output
  // goes to member matrix_data_
  gt::gtensor_device<T, 3> d_A(matrix_data_.shape());
  gt::gtensor_device<T*, 1> d_Aptr(matrix_pointers_.shape());
  detail::init_device_pointer_array(d_Aptr, d_A);
  gt::copy(matrix_data_, d_A);
  gt::synchronize();
  gt::blas::getri_batched<T>(h_, n_, gt::raw_pointer_cast(d_Aptr.data()), n_,
                             gt::raw_pointer_cast(pivot_data_.data()),
                             gt::raw_pointer_cast(matrix_pointers_.data()), n_,
                             gt::raw_pointer_cast(info_.data()), nbatches_);
  gt::synchronize();
}

template <typename T>
void SolverInvert<T>::solve(T* rhs, T* result)
{
  gt::copy_n(gt::device_pointer_cast(rhs), n_ * nrhs_ * nbatches_,
             rhs_input_data_.data());
  gt::blas::gemm_batched<T>(
    h_, n_, nrhs_, n_, 1.0, gt::raw_pointer_cast(matrix_pointers_.data()), n_,
    gt::raw_pointer_cast(rhs_input_pointers_.data()), n_, 0.0,
    gt::raw_pointer_cast(rhs_pointers_.data()), n_, nbatches_);
  gt::copy_n(rhs_data_.data(), n_ * nrhs_ * nbatches_,
             gt::device_pointer_cast(result));
}

template class SolverInvert<float>;
template class SolverInvert<double>;
template class SolverInvert<gt::complex<float>>;
template class SolverInvert<gt::complex<double>>;

#ifdef GTENSOR_DEVICE_SYCL

template <typename T>
SolverDenseSYCL<T>::SolverDenseSYCL(gt::blas::handle_t& h, int n, int nbatches,
                                    int nrhs, T* const* matrix_batches)
  : h_(h),
    n_(n),
    nbatches_(nbatches),
    nrhs_(nrhs),
    matrix_data_(gt::shape(n, n, nbatches)),
    pivot_data_(gt::shape(n, nbatches)),
    rhs_data_(gt::shape(n, nrhs, nbatches)),
    scratch_count_(gt::blas::getrs_strided_batched_scratchpad_size<T>(
      h, n, n, n, nrhs, nbatches)),
    scratch_(scratch_count_)
{
  detail::copy_batch_data(matrix_batches, matrix_data_);
  prepare();
}

template <typename T>
void SolverDenseSYCL<T>::prepare()
{
  auto prep_scratch_count =
    gt::blas::getrf_strided_batched_scratchpad_size<T>(h_, n_, n_, nbatches_);
  gt::space::device_vector<T> prep_scratch(prep_scratch_count);

  gt::blas::getrf_strided_batched<T>(
    h_, n_, gt::raw_pointer_cast(matrix_data_.data()), n_,
    gt::raw_pointer_cast(pivot_data_.data()), nbatches_,
    gt::raw_pointer_cast(prep_scratch.data()), prep_scratch_count);
  // Note: synchronize so it's safe to garbage collect scratch
  gt::synchronize();
}

template <typename T>
void SolverDenseSYCL<T>::solve(T* rhs, T* result)
{
  gt::copy_n(gt::device_pointer_cast(rhs), n_ * nrhs_ * nbatches_,
             rhs_data_.data());
  gt::synchronize();
  gt::blas::getrs_strided_batched<T>(
    h_, n_, nrhs_, gt::raw_pointer_cast(matrix_data_.data()), n_,
    gt::raw_pointer_cast(pivot_data_.data()),
    gt::raw_pointer_cast(rhs_data_.data()), n_, nbatches_,
    gt::raw_pointer_cast(scratch_.data()), scratch_count_);
  gt::synchronize();
  gt::copy_n(rhs_data_.data(), n_ * nrhs_ * nbatches_,
             gt::device_pointer_cast(result));
}

template class SolverDenseSYCL<float>;
template class SolverDenseSYCL<double>;
template class SolverDenseSYCL<gt::complex<float>>;
template class SolverDenseSYCL<gt::complex<double>>;

#endif

} // namespace solver

} // namespace gt
