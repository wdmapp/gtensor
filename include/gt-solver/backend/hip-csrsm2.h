
#ifndef GTENSOR_SOLVER_CUDA_H
#define GTENSOR_SOLVER_CUDA_H

#include <type_traits>

#include "gtensor/gtensor.h"
#include "gtensor/sparse.h"

#include "gt-blas/blas.h"

#include <rocsparse/rocsparse.h>

// ======================================================================
// error handling helper

#define gtSparseCheck(what)                                                    \
  {                                                                            \
    gtSparseCheckImpl(what, __FILE__, __LINE__);                               \
  }

inline void gtSparseCheckImpl(rocsparse_status code, const char* file, int line)
{
  if (code != rocsparse_status_success) {
    fprintf(stderr, "gtSparseCheck: rocsparse status %d at %s:%d\n", code, file,
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
  static constexpr auto buffer_size = rocsparse_dcsrsm_buffer_size;
  static constexpr auto analysis = rocsparse_dcsrsm_analysis;
  static constexpr auto solve = rocsparse_dcsrsm_solve;

  template <typename Ptr>
  static auto cast_pointer(Ptr p)
  {
    return gt::raw_pointer_cast(p);
  }
};

template <>
struct csrsm_functions<float>
{
  static constexpr auto buffer_size = rocsparse_scsrsm_buffer_size;
  static constexpr auto analysis = rocsparse_scsrsm_analysis;
  static constexpr auto solve = rocsparse_scsrsm_solve;

  template <typename Ptr>
  static auto cast_pointer(Ptr p)
  {
    return gt::raw_pointer_cast(p);
  }
};

template <>
struct csrsm_functions<gt::complex<double>>
{
  static constexpr auto buffer_size = rocsparse_zcsrsm_buffer_size;
  static constexpr auto analysis = rocsparse_zcsrsm_analysis;
  static constexpr auto solve = rocsparse_zcsrsm_solve;

  template <typename Ptr>
  static auto cast_pointer(Ptr p)
  {
    using ctype = typename std::conditional<
      std::is_const<typename std::remove_pointer<Ptr>::type>::value,
      const rocsparse_double_complex*, rocsparse_double_complex*>::type;

    return reinterpret_cast<ctype>(gt::raw_pointer_cast(p));
  }
};

template <>
struct csrsm_functions<gt::complex<float>>
{
  static constexpr auto buffer_size = rocsparse_ccsrsm_buffer_size;
  static constexpr auto analysis = rocsparse_ccsrsm_analysis;
  static constexpr auto solve = rocsparse_ccsrsm_solve;

  template <typename Ptr>
  static auto cast_pointer(Ptr p)
  {
    using ctype = typename std::conditional<
      std::is_const<typename std::remove_pointer<Ptr>::type>::value,
      const rocsparse_float_complex*, rocsparse_float_complex*>::type;

    return reinterpret_cast<ctype>(gt::raw_pointer_cast(p));
  }
};

} // namespace detail

class sparse_handle_hip
  : public gt::blas::detail::handle_base<sparse_handle_hip, rocsparse_handle>
{
public:
  sparse_handle_hip() { gtSparseCheck(rocsparse_create_handle(&handle_)); }
  ~sparse_handle_hip() { gtSparseCheck(rocsparse_destroy_handle(handle_)); }

  void set_stream(gt::stream_view sview)
  {
    gtSparseCheck(rocsparse_set_stream(handle_, sview.get_backend_stream()));
  }

  gt::stream_view get_stream()
  {
    hipStream_t s;
    gtSparseCheck(rocsparse_get_stream(handle_, &s));
    return gt::stream_view{s};
  }
};

using sparse_handle_t = sparse_handle_hip;

template <typename T>
class csr_matrix_lu_hip
{
public:
  using value_type = T;
  using space_type = gt::space::device;
  static constexpr bool inplace = true;

  csr_matrix_lu_hip(gt::sparse::csr_matrix<T, space_type>& csr_mat,
                    const T alpha, int nrhs,
                    gt::stream_view sview = gt::stream_view{})
    : csr_mat_(csr_mat), alpha_(alpha), nrhs_(nrhs)
  {
    gtSparseCheck(rocsparse_set_stream(h_.get_backend_handle(),
                                       sview.get_backend_stream()));

    gtSparseCheck(rocsparse_create_mat_descr(&l_desc_));
    gtSparseCheck(
      rocsparse_set_mat_type(l_desc_, rocsparse_matrix_type_general));
    gtSparseCheck(
      rocsparse_set_mat_index_base(l_desc_, rocsparse_index_base_zero));
    gtSparseCheck(
      rocsparse_set_mat_fill_mode(l_desc_, rocsparse_fill_mode_lower));
    gtSparseCheck(
      rocsparse_set_mat_diag_type(l_desc_, rocsparse_diag_type_unit));

    gtSparseCheck(rocsparse_create_mat_descr(&u_desc_));
    gtSparseCheck(
      rocsparse_set_mat_type(u_desc_, rocsparse_matrix_type_general));
    gtSparseCheck(
      rocsparse_set_mat_index_base(u_desc_, rocsparse_index_base_zero));
    gtSparseCheck(
      rocsparse_set_mat_fill_mode(u_desc_, rocsparse_fill_mode_upper));
    gtSparseCheck(
      rocsparse_set_mat_diag_type(u_desc_, rocsparse_diag_type_non_unit));

    gtSparseCheck(rocsparse_create_mat_info(&l_info_));
    gtSparseCheck(rocsparse_create_mat_info(&u_info_));

    gt::gtensor<T, 2, space_type> rhs_tmp(gt::shape(csr_mat.shape(0), nrhs));

    // analyze
    std::size_t l_buf_size, u_buf_size;
    gtSparseCheck(FN::buffer_size(
      h_.get_backend_handle(), rocsparse_operation_none,
      rocsparse_operation_none, csr_mat_.shape(0), nrhs_, csr_mat_.nnz(),
      FN::cast_pointer(&alpha_), l_desc_,
      FN::cast_pointer(csr_mat_.values_data()), csr_mat_.row_ptr_data(),
      csr_mat_.col_ind_data(), FN::cast_pointer(rhs_tmp.data()),
      csr_mat_.shape(0), l_info_, solve_policy_, &l_buf_size));

    gtSparseCheck(FN::buffer_size(
      h_.get_backend_handle(), rocsparse_operation_none,
      rocsparse_operation_none, csr_mat_.shape(0), nrhs_, csr_mat_.nnz(),
      FN::cast_pointer(&alpha_), u_desc_,
      FN::cast_pointer(csr_mat_.values_data()), csr_mat_.row_ptr_data(),
      csr_mat_.col_ind_data(), FN::cast_pointer(rhs_tmp.data()),
      csr_mat_.shape(0), u_info_, solve_policy_, &u_buf_size));

    l_buf_.resize(gt::shape(l_buf_size));
    u_buf_.resize(gt::shape(u_buf_size));

    gtSparseCheck(FN::analysis(
      h_.get_backend_handle(), rocsparse_operation_none,
      rocsparse_operation_none, csr_mat_.shape(0), nrhs_, csr_mat_.nnz(),
      FN::cast_pointer(&alpha_), l_desc_,
      FN::cast_pointer(csr_mat_.values_data()), csr_mat_.row_ptr_data(),
      csr_mat_.col_ind_data(), FN::cast_pointer(rhs_tmp.data()),
      csr_mat_.shape(0), l_info_, analysis_policy_, solve_policy_,
      FN::cast_pointer(l_buf_.data())));

    gtSparseCheck(FN::analysis(
      h_.get_backend_handle(), rocsparse_operation_none,
      rocsparse_operation_none, csr_mat_.shape(0), nrhs_, csr_mat_.nnz(),
      FN::cast_pointer(&alpha_), u_desc_,
      FN::cast_pointer(csr_mat_.values_data()), csr_mat_.row_ptr_data(),
      csr_mat_.col_ind_data(), FN::cast_pointer(rhs_tmp.data()),
      csr_mat_.shape(0), u_info_, analysis_policy_, solve_policy_,
      FN::cast_pointer(u_buf_.data())));
  }

  ~csr_matrix_lu_hip()
  {
    gtSparseCheck(rocsparse_csrsm_clear(h_.get_backend_handle(), l_info_));
    gtSparseCheck(rocsparse_csrsm_clear(h_.get_backend_handle(), u_info_));
  }

  void solve(T* rhs, T* result)
  {
    // in place solve in result vector
    if (result == nullptr) {
      result = rhs;
    } else if (rhs != result) {
      gt::copy_n(gt::device_pointer_cast(rhs), csr_mat_.shape(0) * nrhs_,
                 gt::device_pointer_cast(result));
    }
    gtSparseCheck(FN::solve(
      h_.get_backend_handle(), rocsparse_operation_none,
      rocsparse_operation_none, csr_mat_.shape(0), nrhs_, csr_mat_.nnz(),
      FN::cast_pointer(&alpha_), l_desc_,
      FN::cast_pointer(csr_mat_.values_data()), csr_mat_.row_ptr_data(),
      csr_mat_.col_ind_data(), FN::cast_pointer(result), csr_mat_.shape(0),
      l_info_, solve_policy_, FN::cast_pointer(l_buf_.data())));
    gtSparseCheck(FN::solve(
      h_.get_backend_handle(), rocsparse_operation_none,
      rocsparse_operation_none, csr_mat_.shape(0), nrhs_, csr_mat_.nnz(),
      FN::cast_pointer(&alpha_), u_desc_,
      FN::cast_pointer(csr_mat_.values_data()), csr_mat_.row_ptr_data(),
      csr_mat_.col_ind_data(), FN::cast_pointer(result), csr_mat_.shape(0),
      u_info_, solve_policy_, FN::cast_pointer(u_buf_.data())));
  }

  std::size_t get_device_memory_usage()
  {
    size_t nelements = csr_mat_.nnz() + l_buf_.size() + u_buf_.size();
    size_t nint = csr_mat_.nnz() + csr_mat_.shape(0) + 1;
    return nelements * sizeof(T) + nint * sizeof(int);
  }

private:
  gt::sparse::csr_matrix<T, space_type>& csr_mat_;
  const T alpha_;
  int nrhs_;

  sparse_handle_t h_;
  rocsparse_mat_descr l_desc_;
  rocsparse_mat_descr u_desc_;
  rocsparse_mat_info l_info_;
  rocsparse_mat_info u_info_;
  gt::gtensor_device<T, 1> l_buf_;
  gt::gtensor_device<T, 1> u_buf_;

  const rocsparse_solve_policy solve_policy_ = rocsparse_solve_policy_auto;
  const rocsparse_analysis_policy analysis_policy_ =
    rocsparse_analysis_policy_reuse;

  using FN = detail::csrsm_functions<T>;
};

#define GTENSOR_SOLVER_HAVE_CSR_MATRIX_LU

template <typename T>
using csr_matrix_lu = csr_matrix_lu_hip<T>;

} // namespace solver

} // namespace gt

#endif
