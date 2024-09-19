
#ifndef GTENSOR_SOLVER_CUDA_H
#define GTENSOR_SOLVER_CUDA_H

#include <cstdint>
#include <type_traits>

#include "gtensor/gtensor.h"
#include "gtensor/sparse.h"

#include "gt-blas/blas.h"

#if HIP_VERSION_MAJOR >= 6
#include <rocsparse/rocsparse.h>
#else
#include <rocsparse.h>
#endif

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
  static constexpr rocsparse_datatype dtype = rocsparse_datatype_f64_r;

  template <typename Ptr>
  static auto cast_pointer(Ptr p)
  {
    return gt::raw_pointer_cast(p);
  }
};

template <>
struct csrsm_functions<float>
{
  static constexpr rocsparse_datatype dtype = rocsparse_datatype_f32_r;

  template <typename Ptr>
  static auto cast_pointer(Ptr p)
  {
    return gt::raw_pointer_cast(p);
  }
};

template <>
struct csrsm_functions<gt::complex<double>>
{
  static constexpr rocsparse_datatype dtype = rocsparse_datatype_f64_c;

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
  static constexpr rocsparse_datatype dtype = rocsparse_datatype_f32_c;

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
class csr_matrix_lu_hip_generic
{
public:
  using value_type = T;
  using space_type = gt::space::device;
  static constexpr bool inplace = true;

  csr_matrix_lu_hip_generic(gt::sparse::csr_matrix<T, space_type>& csr_mat,
                            const T alpha, int nrhs,
                            gt::stream_view sview = gt::stream_view{})
    : csr_mat_(csr_mat), alpha_(alpha), nrhs_(nrhs)
  {
    gtSparseCheck(rocsparse_set_stream(h_.get_backend_handle(),
                                       sview.get_backend_stream()));

    gtSparseCheck(rocsparse_create_csr_descr(
      &l_desc_, csr_mat_.shape(0), csr_mat_.shape(1), csr_mat_.nnz(),
      csr_mat_.row_ptr_data(), csr_mat_.col_ind_data(), csr_mat_.values_data(),
      rocsparse_indextype_i32, rocsparse_indextype_i32,
      rocsparse_index_base_zero, FN::dtype));

    rocsparse_fill_mode l_fill_mode = rocsparse_fill_mode_lower;
    rocsparse_diag_type l_diag_type = rocsparse_diag_type_unit;
    gtSparseCheck(rocsparse_spmat_set_attribute(
      l_desc_, rocsparse_spmat_fill_mode, &l_fill_mode, sizeof(l_fill_mode)));
    gtSparseCheck(rocsparse_spmat_set_attribute(
      l_desc_, rocsparse_spmat_diag_type, &l_diag_type, sizeof(l_diag_type)));

    gtSparseCheck(rocsparse_create_csr_descr(
      &u_desc_, csr_mat_.shape(0), csr_mat_.shape(1), csr_mat_.nnz(),
      csr_mat_.row_ptr_data(), csr_mat_.col_ind_data(), csr_mat_.values_data(),
      rocsparse_indextype_i32, rocsparse_indextype_i32,
      rocsparse_index_base_zero, FN::dtype));
    rocsparse_fill_mode u_fill_mode = rocsparse_fill_mode_upper;
    rocsparse_diag_type u_diag_type = rocsparse_diag_type_non_unit;
    gtSparseCheck(rocsparse_spmat_set_attribute(
      u_desc_, rocsparse_spmat_fill_mode, &u_fill_mode, sizeof(u_fill_mode)));
    gtSparseCheck(rocsparse_spmat_set_attribute(
      u_desc_, rocsparse_spmat_diag_type, &u_diag_type, sizeof(u_diag_type)));

    if (nrhs_ > 1) {
      gt::gtensor<T, 2, space_type> rhs_tmp(gt::shape(csr_mat.shape(0), nrhs_));
      gtSparseCheck(rocsparse_create_dnmat_descr(
        &rhs_mat_desc_, csr_mat_.shape(0), nrhs_, csr_mat_.shape(0),
        rhs_tmp.data().get(), FN::dtype, rocsparse_order_column));

      // analyze
      gtSparseCheck(rocsparse_spsm(
        h_.get_backend_handle(), rocsparse_operation_none,
        rocsparse_operation_none, FN::cast_pointer(&alpha_), l_desc_,
        rhs_mat_desc_, rhs_mat_desc_, FN::dtype, rocsparse_spsm_alg_default,
        rocsparse_spsm_stage_buffer_size, &l_buf_size_, nullptr));

      gtSparseCheck(rocsparse_spsm(
        h_.get_backend_handle(), rocsparse_operation_none,
        rocsparse_operation_none, FN::cast_pointer(&alpha_), u_desc_,
        rhs_mat_desc_, rhs_mat_desc_, FN::dtype, rocsparse_spsm_alg_default,
        rocsparse_spsm_stage_buffer_size, &u_buf_size_, nullptr));

      l_buf_.resize(gt::shape(l_buf_size_));
      u_buf_.resize(gt::shape(u_buf_size_));

      gtSparseCheck(rocsparse_spsm(
        h_.get_backend_handle(), rocsparse_operation_none,
        rocsparse_operation_none, FN::cast_pointer(&alpha_), l_desc_,
        rhs_mat_desc_, rhs_mat_desc_, FN::dtype, rocsparse_spsm_alg_default,
        rocsparse_spsm_stage_preprocess, &l_buf_size_,
        FN::cast_pointer(l_buf_.data())));

      gtSparseCheck(rocsparse_spsm(
        h_.get_backend_handle(), rocsparse_operation_none,
        rocsparse_operation_none, FN::cast_pointer(&alpha_), u_desc_,
        rhs_mat_desc_, rhs_mat_desc_, FN::dtype, rocsparse_spsm_alg_default,
        rocsparse_spsm_stage_preprocess, &u_buf_size_,
        FN::cast_pointer(u_buf_.data())));
    } else {
      // Note: spsv APIs may have better performance when nrhs=1

      gt::gtensor<T, 1, space_type> rhs_tmp(gt::shape(csr_mat.shape(0)));
      gtSparseCheck(rocsparse_create_dnvec_descr(
        &rhs_vec_desc_, csr_mat_.shape(0), rhs_tmp.data().get(), FN::dtype));

      // analyze
      gtSparseCheck(rocsparse_spsv(
        h_.get_backend_handle(), rocsparse_operation_none,
        FN::cast_pointer(&alpha_), l_desc_, rhs_vec_desc_, rhs_vec_desc_,
        FN::dtype, rocsparse_spsv_alg_default, rocsparse_spsv_stage_buffer_size,
        &l_buf_size_, nullptr));

      gtSparseCheck(rocsparse_spsv(
        h_.get_backend_handle(), rocsparse_operation_none,
        FN::cast_pointer(&alpha_), u_desc_, rhs_vec_desc_, rhs_vec_desc_,
        FN::dtype, rocsparse_spsv_alg_default, rocsparse_spsv_stage_buffer_size,
        &u_buf_size_, nullptr));

      l_buf_.resize(gt::shape(l_buf_size_));
      u_buf_.resize(gt::shape(u_buf_size_));

      gtSparseCheck(rocsparse_spsv(
        h_.get_backend_handle(), rocsparse_operation_none,
        FN::cast_pointer(&alpha_), l_desc_, rhs_vec_desc_, rhs_vec_desc_,
        FN::dtype, rocsparse_spsv_alg_default, rocsparse_spsv_stage_preprocess,
        &l_buf_size_, FN::cast_pointer(l_buf_.data())));

      gtSparseCheck(rocsparse_spsv(
        h_.get_backend_handle(), rocsparse_operation_none,
        FN::cast_pointer(&alpha_), u_desc_, rhs_vec_desc_, rhs_vec_desc_,
        FN::dtype, rocsparse_spsv_alg_default, rocsparse_spsv_stage_preprocess,
        &u_buf_size_, FN::cast_pointer(u_buf_.data())));
    }
  }

  ~csr_matrix_lu_hip_generic()
  {
    if (nrhs_ > 1) {
      gtSparseCheck(rocsparse_destroy_dnmat_descr(rhs_mat_desc_));
    } else {
      gtSparseCheck(rocsparse_destroy_dnvec_descr(rhs_vec_desc_));
    }
    gtSparseCheck(rocsparse_destroy_spmat_descr(l_desc_));
    gtSparseCheck(rocsparse_destroy_spmat_descr(u_desc_));
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
    if (nrhs_ > 1) {
      gtSparseCheck(rocsparse_dnmat_set_values(rhs_mat_desc_, result));
      gtSparseCheck(
        rocsparse_spsm(h_.get_backend_handle(), rocsparse_operation_none,
                       rocsparse_operation_none, FN::cast_pointer(&alpha_),
                       l_desc_, rhs_mat_desc_, rhs_mat_desc_, FN::dtype,
                       rocsparse_spsm_alg_default, rocsparse_spsm_stage_compute,
                       &l_buf_size_, FN::cast_pointer(l_buf_.data())));
      gtSparseCheck(
        rocsparse_spsm(h_.get_backend_handle(), rocsparse_operation_none,
                       rocsparse_operation_none, FN::cast_pointer(&alpha_),
                       u_desc_, rhs_mat_desc_, rhs_mat_desc_, FN::dtype,
                       rocsparse_spsm_alg_default, rocsparse_spsm_stage_compute,
                       &u_buf_size_, FN::cast_pointer(u_buf_.data())));
    } else {
      gtSparseCheck(rocsparse_dnvec_set_values(rhs_vec_desc_, result));
      gtSparseCheck(rocsparse_spsv(
        h_.get_backend_handle(), rocsparse_operation_none,
        FN::cast_pointer(&alpha_), l_desc_, rhs_vec_desc_, rhs_vec_desc_,
        FN::dtype, rocsparse_spsv_alg_default, rocsparse_spsv_stage_compute,
        &l_buf_size_, FN::cast_pointer(l_buf_.data())));

      gtSparseCheck(rocsparse_spsv(
        h_.get_backend_handle(), rocsparse_operation_none,
        FN::cast_pointer(&alpha_), u_desc_, rhs_vec_desc_, rhs_vec_desc_,
        FN::dtype, rocsparse_spsv_alg_default, rocsparse_spsv_stage_compute,
        &u_buf_size_, FN::cast_pointer(u_buf_.data())));
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
  rocsparse_spmat_descr l_desc_;
  rocsparse_spmat_descr u_desc_;
  rocsparse_dnmat_descr rhs_mat_desc_;
  rocsparse_dnvec_descr rhs_vec_desc_;

  std::size_t l_buf_size_;
  gt::gtensor_device<uint8_t, 1> l_buf_;
  std::size_t u_buf_size_;
  gt::gtensor_device<uint8_t, 1> u_buf_;

  using FN = detail::csrsm_functions<T>;
};

#define GTENSOR_SOLVER_HAVE_CSR_MATRIX_LU

template <typename T>
using csr_matrix_lu = csr_matrix_lu_hip_generic<T>;

} // namespace solver

} // namespace gt

#endif
