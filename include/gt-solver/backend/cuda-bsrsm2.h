
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
  static constexpr auto buffer_size = cusparseDbsrsm2_bufferSize;
  static constexpr auto analysis = cusparseDbsrsm2_analysis;
  static constexpr auto solve = cusparseDbsrsm2_solve;

  template <typename Ptr>
  static auto cast_pointer(Ptr p)
  {
    return gt::raw_pointer_cast(p);
  }
};

template <>
struct csrsm_functions<float>
{
  static constexpr auto buffer_size = cusparseSbsrsm2_bufferSize;
  static constexpr auto analysis = cusparseSbsrsm2_analysis;
  static constexpr auto solve = cusparseSbsrsm2_solve;

  template <typename Ptr>
  static auto cast_pointer(Ptr p)
  {
    return gt::raw_pointer_cast(p);
  }
};

template <>
struct csrsm_functions<gt::complex<double>>
{
  static constexpr auto buffer_size = cusparseZbsrsm2_bufferSize;
  static constexpr auto analysis = cusparseZbsrsm2_analysis;
  static constexpr auto solve = cusparseZbsrsm2_solve;

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
  static constexpr auto buffer_size = cusparseCbsrsm2_bufferSize;
  static constexpr auto analysis = cusparseCbsrsm2_analysis;
  static constexpr auto solve = cusparseCbsrsm2_solve;

  template <typename Ptr>
  static auto cast_pointer(Ptr p)
  {
    using ctype = typename std::conditional<
      std::is_const<typename std::remove_pointer<Ptr>::type>::value,
      const cuComplex*, cuComplex*>::type;

    return reinterpret_cast<ctype>(gt::raw_pointer_cast(p));
  }
};

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
class csr_matrix_lu_cuda_bsrsm2
{
public:
  using value_type = T;
  using space_type = gt::space::device;

  csr_matrix_lu_cuda_bsrsm2(gt::sparse::csr_matrix<T, space_type>& csr_mat,
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

    gtSparseCheck(cusparseCreateBsrsm2Info(&l_info_));
    gtSparseCheck(cusparseCreateBsrsm2Info(&u_info_));

    // analyze
    int l_buf_size, u_buf_size;
    gtSparseCheck(FN::buffer_size(
      h_.get_backend_handle(), CUSPARSE_DIRECTION_COLUMN,
      CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      csr_mat_.shape(0), nrhs_, csr_mat_.nnz(), l_desc_,
      FN::cast_pointer(csr_mat_.values_data()), csr_mat_.row_ptr_data(),
      csr_mat_.col_ind_data(), 1, l_info_, &l_buf_size));

    gtSparseCheck(FN::buffer_size(
      h_.get_backend_handle(), CUSPARSE_DIRECTION_COLUMN,
      CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      csr_mat_.shape(0), nrhs_, csr_mat_.nnz(), u_desc_,
      FN::cast_pointer(csr_mat_.values_data()), csr_mat_.row_ptr_data(),
      csr_mat_.col_ind_data(), 1, u_info_, &u_buf_size));

    l_buf_.resize(gt::shape(l_buf_size));
    u_buf_.resize(gt::shape(u_buf_size));

    gtSparseCheck(FN::analysis(
      h_.get_backend_handle(), CUSPARSE_DIRECTION_COLUMN,
      CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      csr_mat_.shape(0), nrhs_, csr_mat_.nnz(), l_desc_,
      FN::cast_pointer(csr_mat_.values_data()), csr_mat_.row_ptr_data(),
      csr_mat_.col_ind_data(), 1, l_info_, policy_,
      FN::cast_pointer(l_buf_.data())));

    gtSparseCheck(FN::analysis(
      h_.get_backend_handle(), CUSPARSE_DIRECTION_COLUMN,
      CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      csr_mat_.shape(0), nrhs_, csr_mat_.nnz(), u_desc_,
      FN::cast_pointer(csr_mat_.values_data()), csr_mat_.row_ptr_data(),
      csr_mat_.col_ind_data(), 1, u_info_, policy_,
      FN::cast_pointer(u_buf_.data())));
  }

  ~csr_matrix_lu_cuda_bsrsm2()
  {
    cusparseDestroyBsrsm2Info(l_info_);
    cusparseDestroyBsrsm2Info(u_info_);
  }

  void solve(T* rhs, T* result)
  {
    // first solve in place into rhs
    gtSparseCheck(FN::solve(
      h_.get_backend_handle(), CUSPARSE_DIRECTION_COLUMN,
      CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      csr_mat_.shape(0), nrhs_, csr_mat_.nnz(), FN::cast_pointer(&alpha_),
      l_desc_, FN::cast_pointer(csr_mat_.values_data()),
      csr_mat_.row_ptr_data(), csr_mat_.col_ind_data(), 1, l_info_,
      FN::cast_pointer(rhs), csr_mat_.shape(0), FN::cast_pointer(rhs),
      csr_mat_.shape(0), policy_, FN::cast_pointer(l_buf_.data())));
    // second solve uses solution of first as input in rhs, result as output
    gtSparseCheck(FN::solve(
      h_.get_backend_handle(), CUSPARSE_DIRECTION_COLUMN,
      CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
      csr_mat_.shape(0), nrhs_, csr_mat_.nnz(), FN::cast_pointer(&alpha_),
      u_desc_, FN::cast_pointer(csr_mat_.values_data()),
      csr_mat_.row_ptr_data(), csr_mat_.col_ind_data(), 1, u_info_,
      FN::cast_pointer(rhs), csr_mat_.shape(0), FN::cast_pointer(result),
      csr_mat_.shape(0), policy_, FN::cast_pointer(u_buf_.data())));
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
  bsrsm2Info_t l_info_;
  bsrsm2Info_t u_info_;
  gt::gtensor_device<uint8_t, 1> l_buf_;
  gt::gtensor_device<uint8_t, 1> u_buf_;

  const int algo_ = 0; // non-block version
  const cusparseSolvePolicy_t policy_ = CUSPARSE_SOLVE_POLICY_USE_LEVEL;

  using FN = detail::csrsm_functions<T>;
};

template <typename T>
using csr_matrix_lu = csr_matrix_lu_cuda_bsrsm2<T>;

} // namespace solver

} // namespace gt

#endif
