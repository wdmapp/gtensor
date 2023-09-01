
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
  static constexpr cudaDataType dtype = CUDA_R_64F;

  template <typename Ptr>
  static auto cast_pointer(Ptr p)
  {
    return gt::raw_pointer_cast(p);
  }
};

template <>
struct csrsm_functions<float>
{
  static constexpr cudaDataType dtype = CUDA_R_32F;

  template <typename Ptr>
  static auto cast_pointer(Ptr p)
  {
    return gt::raw_pointer_cast(p);
  }
};

template <>
struct csrsm_functions<gt::complex<double>>
{
  static constexpr cudaDataType dtype = CUDA_C_64F;

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
  static constexpr cudaDataType dtype = CUDA_C_32F;

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
class csr_matrix_lu_cuda_generic
{
public:
  using value_type = T;
  using space_type = gt::space::device;
  static constexpr bool inplace = true;

  csr_matrix_lu_cuda_generic(gt::sparse::csr_matrix<T, space_type>& csr_mat,
                             const T alpha, int nrhs,
                             gt::stream_view sview = gt::stream_view{})
    : csr_mat_(csr_mat), alpha_(alpha), nrhs_(nrhs)
  {
    gtSparseCheck(
      cusparseSetStream(h_.get_backend_handle(), sview.get_backend_stream()));

    gtSparseCheck(cusparseCreateCsr(
      &l_desc_, csr_mat_.shape(0), csr_mat_.shape(1), csr_mat_.nnz(),
      csr_mat_.row_ptr_data(), csr_mat_.col_ind_data(), csr_mat_.values_data(),
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
      FN::dtype));
    cusparseFillMode_t l_fill_mode = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t l_diag_type = CUSPARSE_DIAG_TYPE_UNIT;
    gtSparseCheck(cusparseSpMatSetAttribute(l_desc_, CUSPARSE_SPMAT_FILL_MODE,
                                            &l_fill_mode, sizeof(l_fill_mode)));
    gtSparseCheck(cusparseSpMatSetAttribute(l_desc_, CUSPARSE_SPMAT_DIAG_TYPE,
                                            &l_diag_type, sizeof(l_diag_type)));

    gtSparseCheck(cusparseCreateCsr(
      &u_desc_, csr_mat_.shape(0), csr_mat_.shape(1), csr_mat_.nnz(),
      csr_mat_.row_ptr_data(), csr_mat_.col_ind_data(), csr_mat_.values_data(),
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
      FN::dtype));
    cusparseFillMode_t u_fill_mode = CUSPARSE_FILL_MODE_UPPER;
    cusparseDiagType_t u_diag_type = CUSPARSE_DIAG_TYPE_NON_UNIT;
    gtSparseCheck(cusparseSpMatSetAttribute(u_desc_, CUSPARSE_SPMAT_FILL_MODE,
                                            &u_fill_mode, sizeof(u_fill_mode)));
    gtSparseCheck(cusparseSpMatSetAttribute(u_desc_, CUSPARSE_SPMAT_DIAG_TYPE,
                                            &u_diag_type, sizeof(u_diag_type)));

    if (nrhs_ > 1) {
      gtSparseCheck(cusparseSpSM_createDescr(&l_spsm_desc_));
      gtSparseCheck(cusparseSpSM_createDescr(&u_spsm_desc_));

      gt::gtensor<T, 2, space_type> rhs_tmp(gt::shape(csr_mat.shape(0), nrhs_));
      gtSparseCheck(cusparseCreateDnMat(
        &rhs_mat_desc_, csr_mat_.shape(0), nrhs_, csr_mat_.shape(0),
        rhs_tmp.data().get(), FN::dtype, CUSPARSE_ORDER_COL));

      // analyze
      std::size_t l_buf_size, u_buf_size;
      gtSparseCheck(cusparseSpSM_bufferSize(
        h_.get_backend_handle(), CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, FN::cast_pointer(&alpha_), l_desc_,
        rhs_mat_desc_, rhs_mat_desc_, FN::dtype, algo_sm_, l_spsm_desc_,
        &l_buf_size));

      gtSparseCheck(cusparseSpSM_bufferSize(
        h_.get_backend_handle(), CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, FN::cast_pointer(&alpha_), u_desc_,
        rhs_mat_desc_, rhs_mat_desc_, FN::dtype, algo_sm_, u_spsm_desc_,
        &u_buf_size));

      l_buf_.resize(gt::shape(l_buf_size));
      u_buf_.resize(gt::shape(u_buf_size));

      gtSparseCheck(cusparseSpSM_analysis(
        h_.get_backend_handle(), CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, FN::cast_pointer(&alpha_), l_desc_,
        rhs_mat_desc_, rhs_mat_desc_, FN::dtype, algo_sm_, l_spsm_desc_,
        FN::cast_pointer(l_buf_.data())));

      gtSparseCheck(cusparseSpSM_analysis(
        h_.get_backend_handle(), CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, FN::cast_pointer(&alpha_), u_desc_,
        rhs_mat_desc_, rhs_mat_desc_, FN::dtype, algo_sm_, u_spsm_desc_,
        FN::cast_pointer(u_buf_.data())));
    } else {
      // Note: SpSV APIs have better performance when nrhs=1, at least in 12.2
      gtSparseCheck(cusparseSpSV_createDescr(&l_spsv_desc_));
      gtSparseCheck(cusparseSpSV_createDescr(&u_spsv_desc_));

      gt::gtensor<T, 1, space_type> rhs_tmp(gt::shape(csr_mat.shape(0)));
      gtSparseCheck(cusparseCreateDnVec(&rhs_vec_desc_, csr_mat_.shape(0),
                                        rhs_tmp.data().get(), FN::dtype));

      // analyze
      std::size_t l_buf_size, u_buf_size;
      gtSparseCheck(cusparseSpSV_bufferSize(
        h_.get_backend_handle(), CUSPARSE_OPERATION_NON_TRANSPOSE,
        FN::cast_pointer(&alpha_), l_desc_, rhs_vec_desc_, rhs_vec_desc_,
        FN::dtype, algo_sv_, l_spsv_desc_, &l_buf_size));

      gtSparseCheck(cusparseSpSV_bufferSize(
        h_.get_backend_handle(), CUSPARSE_OPERATION_NON_TRANSPOSE,
        FN::cast_pointer(&alpha_), u_desc_, rhs_vec_desc_, rhs_vec_desc_,
        FN::dtype, algo_sv_, u_spsv_desc_, &u_buf_size));

      l_buf_.resize(gt::shape(l_buf_size));
      u_buf_.resize(gt::shape(u_buf_size));

      gtSparseCheck(cusparseSpSV_analysis(
        h_.get_backend_handle(), CUSPARSE_OPERATION_NON_TRANSPOSE,
        FN::cast_pointer(&alpha_), l_desc_, rhs_vec_desc_, rhs_vec_desc_,
        FN::dtype, algo_sv_, l_spsv_desc_, FN::cast_pointer(l_buf_.data())));

      gtSparseCheck(cusparseSpSV_analysis(
        h_.get_backend_handle(), CUSPARSE_OPERATION_NON_TRANSPOSE,
        FN::cast_pointer(&alpha_), u_desc_, rhs_vec_desc_, rhs_vec_desc_,
        FN::dtype, algo_sv_, u_spsv_desc_, FN::cast_pointer(u_buf_.data())));
    }
  }

  ~csr_matrix_lu_cuda_generic()
  {
    if (nrhs_ > 1) {
      gtSparseCheck(cusparseSpSM_destroyDescr(l_spsm_desc_));
      gtSparseCheck(cusparseSpSM_destroyDescr(u_spsm_desc_));
      gtSparseCheck(cusparseDestroyDnMat(rhs_mat_desc_));
    } else {
      gtSparseCheck(cusparseSpSV_destroyDescr(l_spsv_desc_));
      gtSparseCheck(cusparseSpSV_destroyDescr(u_spsv_desc_));
      gtSparseCheck(cusparseDestroyDnVec(rhs_vec_desc_));
    }
    gtSparseCheck(cusparseDestroySpMat(l_desc_));
    gtSparseCheck(cusparseDestroySpMat(u_desc_));
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
      gtSparseCheck(cusparseDnMatSetValues(rhs_mat_desc_, result));
      gtSparseCheck(cusparseSpSM_solve(
        h_.get_backend_handle(), CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, FN::cast_pointer(&alpha_), l_desc_,
        rhs_mat_desc_, rhs_mat_desc_, FN::dtype, algo_sm_, l_spsm_desc_));
      gtSparseCheck(cusparseSpSM_solve(
        h_.get_backend_handle(), CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, FN::cast_pointer(&alpha_), u_desc_,
        rhs_mat_desc_, rhs_mat_desc_, FN::dtype, algo_sm_, u_spsm_desc_));
    } else {
      gtSparseCheck(cusparseDnVecSetValues(rhs_vec_desc_, result));
      gtSparseCheck(cusparseSpSV_solve(
        h_.get_backend_handle(), CUSPARSE_OPERATION_NON_TRANSPOSE,
        FN::cast_pointer(&alpha_), l_desc_, rhs_vec_desc_, rhs_vec_desc_,
        FN::dtype, algo_sv_, l_spsv_desc_));
      gtSparseCheck(cusparseSpSV_solve(
        h_.get_backend_handle(), CUSPARSE_OPERATION_NON_TRANSPOSE,
        FN::cast_pointer(&alpha_), u_desc_, rhs_vec_desc_, rhs_vec_desc_,
        FN::dtype, algo_sv_, u_spsv_desc_));
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
  cusparseSpMatDescr_t l_desc_;
  cusparseSpMatDescr_t u_desc_;
  cusparseSpSMDescr_t l_spsm_desc_;
  cusparseSpSMDescr_t u_spsm_desc_;
  cusparseDnMatDescr_t rhs_mat_desc_;
  cusparseSpSVDescr_t l_spsv_desc_;
  cusparseSpSVDescr_t u_spsv_desc_;
  cusparseDnVecDescr_t rhs_vec_desc_;
  gt::gtensor_device<uint8_t, 1> l_buf_;
  gt::gtensor_device<uint8_t, 1> u_buf_;

  const cusparseSpSMAlg_t algo_sm_ = CUSPARSE_SPSM_ALG_DEFAULT;
  const cusparseSpSVAlg_t algo_sv_ = CUSPARSE_SPSV_ALG_DEFAULT;

  using FN = detail::csrsm_functions<T>;
};

#define GTENSOR_SOLVER_HAVE_CSR_MATRIX_LU

template <typename T>
using csr_matrix_lu = csr_matrix_lu_cuda_generic<T>;

} // namespace solver

} // namespace gt

#endif
