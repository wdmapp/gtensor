#include "gtensor/gtensor.h"

#include "gt-blas/blas.h"

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
    h_ptr_array(i) = gt::raw_pointer_cast(&(d_data_array(0, 0, i)));
  }
  gt::copy(h_ptr_array, d_ptr_array);
}

template <typename HPtrArray, typename DPtrArray, typename T>
void set_device_pointer_array_rhs(HPtrArray& h_ptr_array,
                                  DPtrArray& d_ptr_array, T* rhs, int n,
                                  int nrhs)
{
  assert(h_ptr_array.shape() == d_ptr_array.shape());
  if (h_ptr_array(0) == rhs) {
    // optimize case where same pointer is passed repeatedly
    return;
  }
  int nbatches = d_ptr_array.shape(0);
  int stride = n * nrhs;
  for (int i = 0; i < nbatches; i++) {
    h_ptr_array(i) = rhs + i * stride;
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

#ifdef GTENSOR_DEVICE_SYCL

template <typename T>
solver_dense<T>::solver_dense(gt::blas::handle_t& h, int n, int nbatches,
                              int nrhs, T* const* matrix_batches)
  : h_(h),
    n_(n),
    nbatches_(nbatches),
    nrhs_(nrhs),
    matrix_data_(gt::shape(n, n, nbatches)),
    pivot_data_(gt::shape(n, nbatches)),
    scratch_count_(gt::blas::getrs_strided_batched_scratchpad_size<T>(
      h, n, n, n, nrhs, nbatches)),
    scratch_(scratch_count_)
{
  detail::copy_batch_data(matrix_batches, matrix_data_);

  // factor using strided API
  auto prep_scratch_count =
    gt::blas::getrf_strided_batched_scratchpad_size<T>(h_, n_, n_, nbatches_);
  gt::space::device_vector<T> prep_scratch(prep_scratch_count);

  gt::blas::getrf_strided_batched<T>(
    h_, n_, gt::raw_pointer_cast(matrix_data_.data()), n_,
    gt::raw_pointer_cast(pivot_data_.data()), nbatches_,
    gt::raw_pointer_cast(prep_scratch.data()), prep_scratch_count);
  // Note: synchronize so it's safe to destroy scratch
  gt::synchronize();
}

template <typename T>
void solver_dense<T>::solve(T* rhs, T* result)
{
  gt::blas::getrs_strided_batched<T>(
    h_, n_, nrhs_, gt::raw_pointer_cast(matrix_data_.data()), n_,
    gt::raw_pointer_cast(pivot_data_.data()), rhs, n_, nbatches_,
    gt::raw_pointer_cast(scratch_.data()), scratch_count_);
  if (rhs != result) {
    gt::copy_n(gt::device_pointer_cast(rhs), n_ * nrhs_ * nbatches_,
               gt::device_pointer_cast(result));
  }
}

template <typename T>
std::size_t solver_dense<T>::get_device_memory_usage()
{
  size_t nelements = matrix_data_.size() + scratch_count_;
  size_t nindex = pivot_data_.size();
  return nelements * sizeof(T) + nindex * sizeof(gt::blas::index_t);
}

#else // CUDA and HIP

template <typename T>
solver_dense<T>::solver_dense(gt::blas::handle_t& h, int n, int nbatches,
                              int nrhs, T* const* matrix_batches)
  : h_(h),
    n_(n),
    nbatches_(nbatches),
    nrhs_(nrhs),
    matrix_data_(gt::shape(n, n, nbatches)),
    matrix_pointers_(gt::shape(nbatches)),
    pivot_data_(gt::shape(n, nbatches)),
    info_(gt::shape(nbatches)),
    rhs_pointers_(gt::shape(nbatches)),
    h_rhs_pointers_(gt::shape(nbatches))
{
  // copy non-contiguous host memory to contiguous device memory
  detail::copy_batch_data(matrix_batches, matrix_data_);
  detail::init_device_pointer_array(matrix_pointers_, matrix_data_);

  // dense LU factor with pivot
  gt::blas::getrf_batched<T>(h_, n_,
                             gt::raw_pointer_cast(matrix_pointers_.data()), n_,
                             gt::raw_pointer_cast(pivot_data_.data()),
                             gt::raw_pointer_cast(info_.data()), nbatches_);
  // Note: synchronize for consistency with other implementations
  gt::synchronize();
}

template <typename T>
void solver_dense<T>::solve(T* rhs, T* result)
{
  detail::set_device_pointer_array_rhs(h_rhs_pointers_, rhs_pointers_, rhs, n_,
                                       nrhs_);
  gt::blas::getrs_batched<T>(
    h_, n_, nrhs_, gt::raw_pointer_cast(matrix_pointers_.data()), n_,
    gt::raw_pointer_cast(pivot_data_.data()),
    gt::raw_pointer_cast(rhs_pointers_.data()), n_, nbatches_);
  if (rhs != result && result != nullptr) {
    gt::copy_n(gt::device_pointer_cast(rhs), n_ * nrhs_ * nbatches_,
               gt::device_pointer_cast(result));
  }
}

template <typename T>
std::size_t solver_dense<T>::get_device_memory_usage()
{
  size_t nelements = matrix_data_.size();
  size_t nindex = pivot_data_.size();
  size_t nptr = matrix_pointers_.size() + rhs_pointers_.size();
  return nelements * sizeof(T) + nindex * sizeof(gt::blas::index_t) +
         nptr * sizeof(T*) + info_.size() * sizeof(int);
}

#endif

template class solver_dense<float>;
template class solver_dense<double>;
template class solver_dense<gt::complex<float>>;
template class solver_dense<gt::complex<double>>;

template <typename T>
solver_invert<T>::solver_invert(gt::blas::handle_t& h, int n, int nbatches,
                                int nrhs, T* const* matrix_batches)
  : h_(h),
    n_(n),
    nbatches_(nbatches),
    nrhs_(nrhs),
    matrix_data_(gt::shape(n, n, nbatches)),
    matrix_pointers_(gt::shape(nbatches)),
    pivot_data_(gt::shape(n, nbatches)),
    info_(gt::shape(nbatches)),
    rhs_pointers_(gt::shape(nbatches)),
    h_rhs_pointers_(gt::shape(nbatches)),
    result_pointers_(gt::shape(nbatches)),
    h_result_pointers_(gt::shape(nbatches))
{
  detail::copy_batch_data(matrix_batches, matrix_data_);
  detail::init_device_pointer_array(matrix_pointers_, matrix_data_);

  // LU factor with pivot into matrix_data_
  gt::blas::getrf_batched<T>(h_, n_,
                             gt::raw_pointer_cast(matrix_pointers_.data()), n_,
                             gt::raw_pointer_cast(pivot_data_.data()),
                             gt::raw_pointer_cast(info_.data()), nbatches_);

  // invert using LU factors with getri.
  // Note: getri is not in place, so we copy input data to temporary and have
  // inverted output go into the matrix_data_ member variable
  gt::gtensor_device<T, 3> d_A(matrix_data_.shape());
  gt::gtensor_device<T*, 1> d_Aptr(matrix_pointers_.shape());
  detail::init_device_pointer_array(d_Aptr, d_A);
  gt::copy(matrix_data_, d_A);
  gt::blas::getri_batched<T>(h_, n_, gt::raw_pointer_cast(d_Aptr.data()), n_,
                             gt::raw_pointer_cast(pivot_data_.data()),
                             gt::raw_pointer_cast(matrix_pointers_.data()), n_,
                             gt::raw_pointer_cast(info_.data()), nbatches_);
  // Note: synchronize so it's safe to destroy temporaries
  gt::synchronize();
}

template <typename T>
void solver_invert<T>::solve(T* rhs, T* result)
{
  // in place is not supported
  assert(rhs != result && result != nullptr);
  detail::set_device_pointer_array_rhs(h_rhs_pointers_, rhs_pointers_, rhs, n_,
                                       nrhs_);
  detail::set_device_pointer_array_rhs(h_result_pointers_, result_pointers_,
                                       result, n_, nrhs_);
  gt::blas::gemm_batched<T>(
    h_, n_, nrhs_, n_, 1.0, gt::raw_pointer_cast(matrix_pointers_.data()), n_,
    gt::raw_pointer_cast(rhs_pointers_.data()), n_, 0.0,
    gt::raw_pointer_cast(result_pointers_.data()), n_, nbatches_);
}

template <typename T>
std::size_t solver_invert<T>::get_device_memory_usage()
{
  size_t nelements = matrix_data_.size();
  size_t nindex = pivot_data_.size();
  size_t nptr =
    matrix_pointers_.size() + rhs_pointers_.size() + result_pointers_.size();
  return nelements * sizeof(T) + nindex * sizeof(gt::blas::index_t) +
         nptr * sizeof(T*) + info_.size() * sizeof(int);
}

template class solver_invert<float>;
template class solver_invert<double>;
template class solver_invert<gt::complex<float>>;
template class solver_invert<gt::complex<double>>;

template <typename T>
solver_sparse<T>::solver_sparse(gt::blas::handle_t& blas_h, int n, int nbatches,
                                int nrhs, T* const* matrix_batches)
  : n_(n),
    nbatches_(nbatches),
    nrhs_(nrhs),
    csr_mat_(lu_factor_batches_to_csr(blas_h, n, nbatches, matrix_batches)),
    csr_mat_lu_(csr_mat_, T(1.0), nrhs)
{}

template <typename T>
void solver_sparse<T>::solve(T* rhs, T* result)
{
  csr_mat_lu_.solve(rhs, result);
}

template <typename T>
std::size_t solver_sparse<T>::get_device_memory_usage()
{
  return csr_mat_lu_.get_device_memory_usage();
}

template <typename T>
gt::sparse::csr_matrix<T, gt::space::device>
solver_sparse<T>::lu_factor_batches_to_csr(gt::blas::handle_t& h, int n,
                                           int nbatches,
                                           T* const* matrix_batches)
{
  // temporary arrays to do dense factoring in contiguous device memory
  gt::gtensor_device<T, 3> matrix_data(gt::shape(n, n, nbatches));
  gt::gtensor_device<T*, 1> matrix_pointers(gt::shape(nbatches));
  gt::gtensor_device<int, 1> info(gt::shape(nbatches));

  // copy non-contiguous host memory to contiguous device memory
  detail::copy_batch_data(matrix_batches, matrix_data);
  detail::init_device_pointer_array(matrix_pointers, matrix_data);

  // dense LU factor without pivot
  gt::blas::getrf_npvt_batched<T>(
    h, n, gt::raw_pointer_cast(matrix_pointers.data()), n,
    gt::raw_pointer_cast(info.data()), nbatches);
  gt::synchronize();

  // convert to single sparse CSR format matrix, with each batch matrix
  // along the diagonal
  return gt::sparse::csr_matrix<T, gt::space::device>::join_matrix_batches(
    matrix_data);
}

template class solver_sparse<float>;
template class solver_sparse<double>;

// Note: oneMKL sparse API does not support complex yet
#if defined(GTENSOR_DEVICE_CUDA) || defined(GTENSOR_DEVICE_HIP)
template class solver_sparse<gt::complex<float>>;
template class solver_sparse<gt::complex<double>>;
#endif

template <typename T>
solver_banded<T>::solver_banded(gt::blas::handle_t& h, int n, int nbatches,
                                int nrhs, T* const* matrix_batches)
  : h_(h),
    n_(n),
    nbatches_(nbatches),
    nrhs_(nrhs),
    matrix_data_(gt::shape(n, n, nbatches)),
    matrix_pointers_(gt::shape(nbatches)),
    pivot_data_(gt::shape(n, nbatches)),
    info_(gt::shape(nbatches)),
    rhs_pointers_(gt::shape(nbatches)),
    h_rhs_pointers_(gt::shape(nbatches))
{
  // copy non-contiguous host memory to contiguous device memory
  detail::copy_batch_data(matrix_batches, matrix_data_);
  detail::init_device_pointer_array(matrix_pointers_, matrix_data_);

  // band LU factor with pivot
  gt::blas::getrf_batched<T>(h_, n_,
                             gt::raw_pointer_cast(matrix_pointers_.data()), n_,
                             gt::raw_pointer_cast(pivot_data_.data()),
                             gt::raw_pointer_cast(info_.data()), nbatches_);
  // Note: synchronize for consistency with other implementations
  gt::synchronize();

  auto bw = gt::blas::get_max_bandwidth(
    h_, n_, gt::raw_pointer_cast(matrix_pointers_.data()), n_, nbatches_);
  lbw_ = bw.lower;
  ubw_ = bw.upper;
}

template <typename T>
void solver_banded<T>::solve(T* rhs, T* result)
{
  detail::set_device_pointer_array_rhs(h_rhs_pointers_, rhs_pointers_, rhs, n_,
                                       nrhs_);
  gt::blas::getrs_banded_batched<T>(
    h_, n_, nrhs_, gt::raw_pointer_cast(matrix_pointers_.data()), n_,
    gt::raw_pointer_cast(pivot_data_.data()),
    gt::raw_pointer_cast(rhs_pointers_.data()), n_, nbatches_, lbw_, ubw_);
  if (rhs != result && result != nullptr) {
    gt::copy_n(gt::device_pointer_cast(rhs), n_ * nrhs_ * nbatches_,
               gt::device_pointer_cast(result));
  }
}

template <typename T>
std::size_t solver_banded<T>::get_device_memory_usage()
{
  size_t nelements = matrix_data_.size();
  size_t nindex = pivot_data_.size();
  size_t nptr = matrix_pointers_.size() + rhs_pointers_.size();
  return nelements * sizeof(T) + nindex * sizeof(gt::blas::index_t) +
         nptr * sizeof(T*) + info_.size() * sizeof(int);
}

template class solver_banded<float>;
template class solver_banded<double>;
template class solver_banded<gt::complex<float>>;
template class solver_banded<gt::complex<double>>;

template <typename Solver>
staging_solver<Solver>::staging_solver(gt::blas::handle_t& h, int n,
                                       int nbatches, int nrhs,
                                       value_type* const* matrix_batches)
  : n_(n),
    nbatches_(nbatches),
    nrhs_(nrhs),
    solver_(h, n, nbatches, nrhs, matrix_batches),
    rhs_stage_(gt::shape(n, nrhs, nbatches)),
    rhs_stage_p_(gt::raw_pointer_cast(rhs_stage_.data())),
    result_stage_(Solver::inplace ? gt::shape(0, 0, 0)
                                  : gt::shape(n, nrhs, nbatches)),
    result_stage_p_(
      Solver::inplace ? nullptr : gt::raw_pointer_cast(result_stage_.data()))
{}

template <typename Solver>
void staging_solver<Solver>::solve(value_type* rhs, value_type* result)
{
  gt::copy_n(gt::device_pointer_cast(rhs), rhs_stage_.size(),
             rhs_stage_.data());
  if (inplace) {
    solver_.solve(rhs_stage_p_, result);
  } else {
    solver_.solve(rhs_stage_p_, result_stage_p_);
  }
  if (!inplace) {
    gt::copy_n(result_stage_.data(), result_stage_.size(),
               gt::device_pointer_cast(result));
  }
}

template <typename Solver>
std::size_t staging_solver<Solver>::get_device_memory_usage()
{
  size_t nelements = rhs_stage_.size() + result_stage_.size();
  return solver_.get_device_memory_usage() + nelements * sizeof(value_type);
}

template class staging_solver<solver_dense<float>>;
template class staging_solver<solver_dense<double>>;
template class staging_solver<solver_dense<gt::complex<float>>>;
template class staging_solver<solver_dense<gt::complex<double>>>;

template class staging_solver<solver_invert<float>>;
template class staging_solver<solver_invert<double>>;
template class staging_solver<solver_invert<gt::complex<float>>>;
template class staging_solver<solver_invert<gt::complex<double>>>;

template class staging_solver<solver_sparse<float>>;
template class staging_solver<solver_sparse<double>>;
// Note: oneMKL sparse API does not support complex yet
#if defined(GTENSOR_DEVICE_CUDA) || defined(GTENSOR_DEVICE_HIP)
template class staging_solver<solver_sparse<gt::complex<float>>>;
template class staging_solver<solver_sparse<gt::complex<double>>>;
#endif

template class staging_solver<solver_banded<float>>;
template class staging_solver<solver_banded<double>>;
template class staging_solver<solver_banded<gt::complex<float>>>;
template class staging_solver<solver_banded<gt::complex<double>>>;

} // namespace solver

} // namespace gt
