
#ifndef GTENSOR_SOLVER_CUDA_H
#define GTENSOR_SOLVER_CUDA_H

#include <cstdint>
#include <type_traits>

#include "gtensor/gtensor.h"
#include "gtensor/sparse.h"

#include "gt-blas/blas.h"

#include <cusparse.h>

// ======================================================================
// error handling helper

#define gtSparseCheck(what)                                                    \
  {                                                                            \
    gtSparseCheckImpl(what, __FILE__, __LINE__);                               \
  }

inline void gtSparseCheckImpl(cusparseStatus_t code, const char* file, int line)
{
  if (code != CUSPARSE_STATUS_SUCCESS) {
    fprintf(stderr, "gtSparseCheck: cusparse status %d at %s:%d\n", code, file,
            line);
    abort();
  }
}

namespace gt
{

namespace solver
{

namespace detail
{

template <typename T>
struct csrsm_functions;

template <>
struct csrsm_functions<double>
{
  static constexpr auto buffer_size = cusparseDcsrsm2_bufferSizeExt;
  static constexpr auto analysis = cusparseDcsrsm2_analysis;
  static constexpr auto solve = cusparseDcsrsm2_solve;

  template <typename Ptr>
  static auto cast_pointer(Ptr p)
  {
    return gt::raw_pointer_cast(p);
  }
};

template <>
struct csrsm_functions<float>
{
  static constexpr auto buffer_size = cusparseScsrsm2_bufferSizeExt;
  static constexpr auto analysis = cusparseScsrsm2_analysis;
  static constexpr auto solve = cusparseScsrsm2_solve;

  template <typename Ptr>
  static auto cast_pointer(Ptr p)
  {
    return gt::raw_pointer_cast(p);
  }
};

template <>
struct csrsm_functions<gt::complex<double>>
{
  static constexpr auto buffer_size = cusparseZcsrsm2_bufferSizeExt;
  static constexpr auto analysis = cusparseZcsrsm2_analysis;
  static constexpr auto solve = cusparseZcsrsm2_solve;

  template <typename Ptr>
  static auto cast_pointer(Ptr p)
  {
    using ctype = typename std::conditional<
      std::is_const<typename std::remove_pointer<Ptr>::type>::value,
      const cuDoubleComplex*, cuDoubleComplex*>::type;

    return reinterpret_cast<ctype>(gt::raw_pointer_cast(p));
  }
};

template <>
struct csrsm_functions<gt::complex<float>>
{
  static constexpr auto buffer_size = cusparseCcsrsm2_bufferSizeExt;
  static constexpr auto analysis = cusparseCcsrsm2_analysis;
  static constexpr auto solve = cusparseCcsrsm2_solve;

  template <typename Ptr>
  static auto cast_pointer(Ptr p)
  {
    using ctype = typename std::conditional<
      std::is_const<typename std::remove_pointer<Ptr>::type>::value,
      const cuComplex*, cuComplex*>::type;

    return reinterpret_cast<ctype>(gt::raw_pointer_cast(p));
  }
};

/*
template <typename T, typename S>
void csrsm_buffer_size(sparse_handle_t& h, int algorithm, csr_matrix<T, S>& mat,
                       T alpha, cusparseMatDesc_t l_desc, cusparseMatDesc
    gtSparseCheck(detail::csrsm_functions<T>::analysis(
      h_.get_backend_handle(), algo_, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, csr_mat_.shape(0), nrhs_,
csr_mat_.nnz(), &alpha_, l_desc_,
      detail::csrsm_functions<T>::cast_pointer(csr_mat_.values_data()),
      csr_mat_.row_ptr_data(), csr_mat_.col_ind_data(), rhs_tmp.data(), 1,
l_info_, detail::csrsm_functions<T>::cast_pointer(buf_.data())));
      */

} // namespace detail

class sparse_handle_cuda
  : public gt::blas::detail::handle_base<sparse_handle_cuda, cusparseHandle_t>
{
public:
  sparse_handle_cuda() { gtSparseCheck(cusparseCreate(&handle_)); }
  ~sparse_handle_cuda() { gtSparseCheck(cusparseDestroy(handle_)); }

  void set_stream(gt::stream_view sview)
  {
    gtSparseCheck(cusparseSetStream(handle_, sview.get_backend_stream()));
  }

  gt::stream_view get_stream()
  {
    cudaStream_t s;
    gtSparseCheck(cusparseGetStream(handle_, &s));
    return gt::stream_view{s};
  }
};

using sparse_handle_t = sparse_handle_cuda;

template <typename T>
class csr_matrix_lu_cuda_csrsm2
{
public:
  using value_type = T;
  using space_type = gt::space::device;

  csr_matrix_lu_cuda_csrsm2(gt::sparse::csr_matrix<T, space_type>& csr_mat,
                            const T alpha, int nrhs,
                            gt::stream_view sview = gt::stream_view{})
    : csr_mat_(csr_mat), alpha_(alpha), nrhs_(nrhs)
  {
    gtSparseCheck(
      cusparseSetStream(h_.get_backend_handle(), sview.get_backend_stream()));

    gtSparseCheck(cusparseCreateMatDescr(&l_desc_));
    gtSparseCheck(cusparseSetMatType(l_desc_, CUSPARSE_MATRIX_TYPE_GENERAL));
    gtSparseCheck(cusparseSetMatIndexBase(l_desc_, CUSPARSE_INDEX_BASE_ZERO));
    gtSparseCheck(cusparseSetMatFillMode(l_desc_, CUSPARSE_FILL_MODE_LOWER));
    gtSparseCheck(cusparseSetMatDiagType(l_desc_, CUSPARSE_DIAG_TYPE_UNIT));

    gtSparseCheck(cusparseCreateMatDescr(&u_desc_));
    gtSparseCheck(cusparseSetMatType(u_desc_, CUSPARSE_MATRIX_TYPE_GENERAL));
    gtSparseCheck(cusparseSetMatIndexBase(u_desc_, CUSPARSE_INDEX_BASE_ZERO));
    gtSparseCheck(cusparseSetMatFillMode(u_desc_, CUSPARSE_FILL_MODE_UPPER));
    gtSparseCheck(cusparseSetMatDiagType(u_desc_, CUSPARSE_DIAG_TYPE_NON_UNIT));

    gtSparseCheck(cusparseCreateCsrsm2Info(&l_info_));
    gtSparseCheck(cusparseCreateCsrsm2Info(&u_info_));

    gt::gtensor<T, 2, space_type> rhs_tmp(gt::shape(csr_mat.shape(0), nrhs));

    // analyze
    std::size_t l_buf_size, u_buf_size;
    gtSparseCheck(FN::buffer_size(
      h_.get_backend_handle(), algo_, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, csr_mat_.shape(0), nrhs_,
      csr_mat_.nnz(), FN::cast_pointer(&alpha_), l_desc_,
      FN::cast_pointer(csr_mat_.values_data()), csr_mat_.row_ptr_data(),
      csr_mat_.col_ind_data(), FN::cast_pointer(rhs_tmp.data()),
      csr_mat_.shape(0), l_info_, policy_, &l_buf_size));

    gtSparseCheck(FN::buffer_size(
      h_.get_backend_handle(), algo_, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, csr_mat_.shape(0), nrhs_,
      csr_mat_.nnz(), FN::cast_pointer(&alpha_), u_desc_,
      FN::cast_pointer(csr_mat_.values_data()), csr_mat_.row_ptr_data(),
      csr_mat_.col_ind_data(), FN::cast_pointer(rhs_tmp.data()),
      csr_mat_.shape(0), u_info_, policy_, &u_buf_size));

    l_buf_.resize(gt::shape(l_buf_size));
    u_buf_.resize(gt::shape(u_buf_size));

    gtSparseCheck(FN::analysis(
      h_.get_backend_handle(), algo_, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, csr_mat_.shape(0), nrhs_,
      csr_mat_.nnz(), FN::cast_pointer(&alpha_), l_desc_,
      FN::cast_pointer(csr_mat_.values_data()), csr_mat_.row_ptr_data(),
      csr_mat_.col_ind_data(), FN::cast_pointer(rhs_tmp.data()),
      csr_mat_.shape(0), l_info_, policy_, FN::cast_pointer(l_buf_.data())));

    gtSparseCheck(FN::analysis(
      h_.get_backend_handle(), algo_, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, csr_mat_.shape(0), nrhs_,
      csr_mat_.nnz(), FN::cast_pointer(&alpha_), u_desc_,
      FN::cast_pointer(csr_mat_.values_data()), csr_mat_.row_ptr_data(),
      csr_mat_.col_ind_data(), FN::cast_pointer(rhs_tmp.data()),
      csr_mat_.shape(0), u_info_, policy_, FN::cast_pointer(u_buf_.data())));
  }

  ~csr_matrix_lu_cuda_csrsm2()
  {
    cusparseDestroyCsrsm2Info(l_info_);
    cusparseDestroyCsrsm2Info(u_info_);
  }

  void solve(T* rhs, T* result)
  {
    gtSparseCheck(FN::solve(
      h_.get_backend_handle(), algo_, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, csr_mat_.shape(0), nrhs_,
      csr_mat_.nnz(), FN::cast_pointer(&alpha_), l_desc_,
      FN::cast_pointer(csr_mat_.values_data()), csr_mat_.row_ptr_data(),
      csr_mat_.col_ind_data(), FN::cast_pointer(rhs), csr_mat_.shape(0),
      l_info_, policy_, FN::cast_pointer(l_buf_.data())));
    gtSparseCheck(FN::solve(
      h_.get_backend_handle(), algo_, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, csr_mat_.shape(0), nrhs_,
      csr_mat_.nnz(), FN::cast_pointer(&alpha_), u_desc_,
      FN::cast_pointer(csr_mat_.values_data()), csr_mat_.row_ptr_data(),
      csr_mat_.col_ind_data(), FN::cast_pointer(rhs), csr_mat_.shape(0),
      u_info_, policy_, FN::cast_pointer(u_buf_.data())));

    if (rhs != result && result != nullptr) {
      gt::copy_n(gt::device_pointer_cast(rhs), csr_mat_.shape(0) * nrhs_,
                 gt::device_pointer_cast(result));
    }
  }

  std::size_t get_device_memory_usage()
  {
    size_t nelements = csr_mat_.nnz();
    size_t nbuf = l_buf_.size() + u_buf_.size();
    size_t nint = csr_mat_.nnz() + csr_mat_.shape(0) + 1;
    return nelements * sizeof(T) + nint * sizeof(int) + nbuf;
  }

private:
  gt::sparse::csr_matrix<T, space_type>& csr_mat_;
  const T alpha_;
  int nrhs_;

  sparse_handle_t h_;
  cusparseMatDescr_t l_desc_;
  cusparseMatDescr_t u_desc_;
  csrsm2Info_t l_info_;
  csrsm2Info_t u_info_;
  gt::gtensor_device<uint8_t, 1> l_buf_;
  gt::gtensor_device<uint8_t, 1> u_buf_;

  const int algo_ = 0; // non-block version
  const cusparseSolvePolicy_t policy_ = CUSPARSE_SOLVE_POLICY_USE_LEVEL;

  using FN = detail::csrsm_functions<T>;
};

template <typename T>
using csr_matrix_lu = csr_matrix_lu_cuda_csrsm2<T>;

} // namespace solver

} // namespace gt

#endif
