
#ifndef GTENSOR_SOLVER_CUDA_H
#define GTENSOR_SOLVER_CUDA_H

#include <type_traits>

#include "gtensor/gtensor.h"
#include "gtensor/sparse.h"

#include "gt-blas/blas.h"

#include "oneapi/mkl.hpp"

using namespace gt::complex_cast;

namespace gt
{

namespace solver
{

class sparse_handle_sycl
  : public gt::blas::detail::handle_base<sparse_handle_sycl, sycl::queue>
{
public:
  sparse_handle_sycl() { handle_ = gt::backend::sycl::get_queue(); }

  void set_stream(gt::stream_view sview)
  {
    handle_ = sview.get_backend_stream();
  }

  gt::stream_view get_stream() { return gt::stream_view{handle_}; }
};

using sparse_handle_t = sparse_handle_sycl;

template <typename T>
class csr_matrix_lu_sycl
{
public:
  using value_type = T;
  using space_type = gt::space::device;

  csr_matrix_lu_sycl(gt::sparse::csr_matrix<T, space_type>& csr_mat,
                     const T alpha, int nrhs,
                     gt::stream_view sview = gt::stream_view{})
    : csr_mat_(csr_mat),
      alpha_(alpha),
      nrhs_(nrhs),
      q_(sview.get_backend_stream()),
      rhs_tmp_(gt::shape(csr_mat.shape(0), nrhs)),
      result_tmp_(gt::shape(csr_mat.shape(0), nrhs))
  {
    oneapi::mkl::sparse::init_matrix_handle(&mat_h_);
    auto e = oneapi::mkl::sparse::set_csr_data(
      q_, mat_h_, csr_mat_.shape(0), csr_mat_.shape(1),
      oneapi::mkl::index_base::zero, csr_mat_.row_ptr_data(),
      csr_mat_.col_ind_data(), std_cast(csr_mat_.values_data()));
    e.wait();
  }

  ~csr_matrix_lu_sycl()
  {
    // destroy matrix handle??
  }

  void solve(T* rhs, T* result)
  {
    gt::copy_n(gt::device_pointer_cast(rhs), result_tmp_.size(),
               result_tmp_.data());
    // Note: this oneAPI MKL sparse API does not support multiple right
    // hand sides directly, so instead loop. Since we use in-order queues
    // to make behavior more like CUDA/HIP, this will likely not run each
    // rhs in parallel and not perform very well for multi-rhs solves.
    for (int irhs = 0; irhs < nrhs_; irhs++) {
      auto result_ptr =
        std_cast(result_tmp_.data() + irhs * result_tmp_.shape(0));
      auto rhs_ptr = std_cast(rhs_tmp_.data() + irhs * rhs_tmp_.shape(0));
      auto e = oneapi::mkl::sparse::trsv(
        q_, oneapi::mkl::uplo::lower, oneapi::mkl::transpose::nontrans,
        oneapi::mkl::diag::unit, mat_h_, result_ptr, rhs_ptr);
      oneapi::mkl::sparse::trsv(
        q_, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans,
        oneapi::mkl::diag::nonunit, mat_h_, rhs_ptr, result_ptr);
    }
    gt::copy_n(result_tmp_.data(), result_tmp_.size(),
               gt::device_pointer_cast(result));
  }

  std::size_t get_device_memory_usage()
  {
    size_t nelements = csr_mat_.nnz() + rhs_tmp_.size() + result_tmp_.size() +
                       l_buf_.size() + u_buf_.size();
    size_t nint = csr_mat_.nnz() + csr_mat_.shape(0) + 1;
    return nelements * sizeof(T) + nint * sizeof(int);
  }

private:
  gt::sparse::csr_matrix<T, space_type>& csr_mat_;
  T alpha_;
  int nrhs_;
  sparse_handle_t h_;
  sycl::queue& q_;
  gt::gtensor<T, 2, space_type> rhs_tmp_;
  gt::gtensor<T, 2, space_type> result_tmp_;
  oneapi::mkl::sparse::matrix_handle_t mat_h_;
};

template <typename T>
using csr_matrix_lu = csr_matrix_lu_sycl<T>;

} // namespace solver

} // namespace gt

#endif
