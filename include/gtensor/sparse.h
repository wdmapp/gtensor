#ifndef GTENSOR_SPARSE_H
#define GTENSOR_SPARSE_H

#include <numeric>
#include <type_traits>

#include "gtensor.h"
#include "reductions.h"

namespace gt
{

namespace sparse
{

namespace detail
{

template <typename DataArray>
gt::gtensor<int, 1> row_ptr_batches(DataArray& d_a_batches, int nbatches)
{
  using S = typename DataArray::space_type;
  using T = typename DataArray::value_type;

  int nrows = d_a_batches.shape(0);
  int ncols = d_a_batches.shape(1);
  gt::gtensor<int, 2, S> d_row_nnz_counts{gt::shape(nrows, nbatches)};
  gt::gtensor<int, 2> h_row_nnz_counts{d_row_nnz_counts.shape()};
  gt::gtensor<int, 1> h_row_ptr{gt::shape(nrows * nbatches + 1)};

  auto k_row_nnz_counts = d_row_nnz_counts.to_kernel();
  auto k_a_batches = d_a_batches.to_kernel();
  gt::launch<2, S>(
    d_row_nnz_counts.shape(), GT_LAMBDA(int i, int b) {
      int nnz = 0;
      for (int j = 0; j < ncols; j++) {
        // Note: casting to T on lhs is needed for thrust backends because the
        // thrust device reference object does not compare properly
        if (T(k_a_batches(i, j, b)) != T(0)) {
          nnz++;
        }
      }
      k_row_nnz_counts(i, b) = nnz;
    });
  gt::copy(d_row_nnz_counts, h_row_nnz_counts);

  // TODO: add scan or partial sum to reductions, keep on device?
  h_row_ptr(0) = 0;
  std::partial_sum(h_row_nnz_counts.data(),
                   h_row_nnz_counts.data() + nrows * nbatches,
                   h_row_ptr.data() + 1);
  return h_row_ptr;
}

} // namespace detail

template <typename T, typename S>
class csr_matrix_span : public expression<csr_matrix_span<T, S>>
{
public:
  using value_type = T;
  using space_type = S;
  using pointer = typename std::add_pointer<value_type>::type;
  using const_pointer = typename std::add_const<pointer>::type;
  using reference = typename std::add_lvalue_reference<value_type>::type;
  using const_reference = typename std::add_const<reference>::type;
  using shape_type = gt::shape_type<2>;

  csr_matrix_span(gt::shape_type<2> shape, int nnz,
                  gt::gtensor_span<T, 1, S> values,
                  gt::gtensor_span<int, 1, S> col_ind,
                  gt::gtensor_span<int, 1, S> row_ptr)
    : shape_(shape),
      nnz_(nnz),
      values_(values),
      col_ind_(col_ind),
      row_ptr_(row_ptr)
  {}

  constexpr static size_type dimension() { return 2; }

  GT_INLINE reference operator[](std::size_t idx) const { return values_(idx); }

  GT_INLINE value_type operator()(std::size_t i, std::size_t j) const
  {
    int idx = values_index(i, j);
    if (idx == -1) {
      return 0;
    }
    return values_(idx);
  }

  GT_INLINE int values_index(std::size_t i, std::size_t j) const
  {
    int row_start = row_ptr_[i];
    int row_end = row_ptr_[i + 1];
    for (int idx = row_start; idx < row_end; idx++) {
      int col_ind = col_ind_(idx);
      if (col_ind == j) {
        return idx;
      } else if (col_ind > j) {
        break;
      }
    }
    return -1;
  }

  GT_INLINE int row_ptr(std::size_t i) const { return row_ptr_(i); }

  GT_INLINE int col_ind(std::size_t i) const { return col_ind_(i); }

  GT_INLINE int nnz() const { return nnz_; }
  GT_INLINE auto size() const { return calc_size(shape_); }
  GT_INLINE auto shape() const { return shape_; }
  GT_INLINE auto shape(int i) const { return shape_[i]; }

  inline auto to_kernel() const { return *this; }

private:
  shape_type shape_;
  int nnz_;
  gt::gtensor_span<value_type, 1, S> values_;
  gt::gtensor_span<int, 1, S> row_ptr_;
  gt::gtensor_span<int, 1, S> col_ind_;
};

template <typename T, typename S>
class csr_matrix : public expression<csr_matrix<T, S>>
{
public:
  using value_type = T;
  using space_type = S;
  using pointer = typename std::add_pointer<value_type>::type;
  using const_pointer = typename std::add_const<pointer>::type;
  using reference = typename std::add_lvalue_reference<value_type>::type;
  using const_reference = typename std::add_const<reference>::type;
  using shape_type = gt::shape_type<2>;
  using kernel_type = csr_matrix_span<value_type, space_type>;
  using const_kernel_type =
    csr_matrix_span<std::add_const_t<value_type>, space_type>;

  csr_matrix(gt::shape_type<2> shape, int nnz) : shape_(shape), nnz_(nnz)
  {
    values_.resize({nnz_});
    col_ind_.resize({nnz_});
    row_ptr_.resize({shape[0] + 1});
  }

  template <typename MatrixType>
  csr_matrix(MatrixType& d_a)
  {
    static_assert(expr_dimension<MatrixType>() == 2,
                  "non-batched sparse construction requires a 2d object");
    shape_ = d_a.shape();

    auto d_batches_view = d_a.view(gt::all, gt::all, gt::newaxis);
    auto h_row_ptr = detail::row_ptr_batches(d_batches_view, 1);
    nnz_ = h_row_ptr(shape_[0]);

    values_.resize({nnz_});
    col_ind_.resize({nnz_});
    row_ptr_.resize({shape_[0] + 1});

    gt::copy(h_row_ptr, row_ptr_);

    convert_batches(d_batches_view, row_ptr_);
  }

  template <typename BatchData>
  static auto join_matrix_batches(BatchData& d_matrix_batches)
  {
    static_assert(expr_dimension<BatchData>() == 3,
                  "batched sparse construction requires a 3d object");
    int nrows = d_matrix_batches.shape(0);
    int ncols = d_matrix_batches.shape(1);
    int nbatches = d_matrix_batches.shape(2);
    /*
    auto h_nnz_offsets = detail::batch_nnz_offsets(d_matrix_batches, nbatches);
    gt::gtensor<int, 1, S> d_nnz_offsets{h_nnz_offsets.shape()};
    csr_matrix csr_mat(gt::shape(nrows * nbatches, ncols * nbatches),
                       h_nnz_offsets(nbatches));

    gt::copy(h_nnz_offsets, d_nnz_offsets);

    csr_mat.convert_batches(d_matrix_batches, d_nnz_offsets);
    return csr_mat;
    */
    auto h_row_ptr = detail::row_ptr_batches(d_matrix_batches, nbatches);
    csr_matrix csr_mat(gt::shape(nrows * nbatches, ncols * nbatches),
                       h_row_ptr(nrows * nbatches));
    gt::copy(h_row_ptr, csr_mat.row_ptr_);

    csr_mat.convert_batches(d_matrix_batches, csr_mat.row_ptr_);
    return csr_mat;
  }

  constexpr static size_type dimension() { return 2; }

  template <typename BatchView>
  void convert_batches(BatchView& d_matrix_view,
                       gt::gtensor<int, 1, S>& d_row_ptr)
  {
    static_assert(expr_dimension<BatchView>() == 3,
                  "3d view required for common dense conversion helper");
    auto k_row_ptr = d_row_ptr.to_kernel();
    auto k_matrix_view = d_matrix_view.to_kernel();
    int nrows = d_matrix_view.shape(0);
    int ncols = d_matrix_view.shape(1);
    int nbatches = d_matrix_view.shape(2);

    auto k_values_ = values_.to_kernel();
    auto k_col_ind_ = col_ind_.to_kernel();

    // past all matrices along diagonal of sparse matrix
    gt::launch<2, S>(
      gt::shape(nrows, nbatches), GT_LAMBDA(int i, int b) {
        int value_offset = k_row_ptr(i + b * nrows);
        int col_offset = ncols * b;
        T temp;
        // Note: we are doing a transpose, since CSR is row major
        for (int j = 0; j < ncols; j++) {
          temp = k_matrix_view(i, j, b);
          if (temp != T(0)) {
            k_values_(value_offset) = temp;
            k_col_ind_(value_offset) = col_offset + j;
            value_offset++;
          }
        }
      });
  }

  GT_INLINE reference operator[](std::size_t idx) const { return values_(idx); }

  GT_INLINE value_type operator()(std::size_t i, std::size_t j) const
  {
    int idx = values_index(i, j);
    if (idx == -1) {
      return 0;
    }
    return values_(idx);
  }

  GT_INLINE int values_index(std::size_t i, std::size_t j) const
  {
    int row_start = row_ptr_[i];
    int row_end = row_ptr_[i + 1];
    for (int idx = row_start; idx < row_end; idx++) {
      int col_ind = col_ind_(idx);
      if (col_ind == j) {
        return idx;
      } else if (col_ind > j) {
        break;
      }
    }
    return -1;
  }

  GT_INLINE int row_ptr(std::size_t i) const { return row_ptr_(i); }

  GT_INLINE int col_ind(std::size_t i) const { return col_ind_(i); }

  GT_INLINE int nnz() const { return nnz_; }
  GT_INLINE auto size() const { return calc_size(shape_); }
  GT_INLINE auto shape() const { return shape_; }
  GT_INLINE auto shape(int i) const { return shape_[i]; }

  inline auto to_kernel() const
  {
    return const_kernel_type(shape_, nnz_, values_.to_kernel(),
                             col_ind_.to_kernel(), row_ptr_.to_kernel());
  }

  inline auto to_kernel()
  {
    return kernel_type(shape_, nnz_, values_.to_kernel(), col_ind_.to_kernel(),
                       row_ptr_.to_kernel());
  }

private:
  int nnz_;
  shape_type shape_;
  gt::gtensor<T, 1, S> values_;
  gt::gtensor<int, 1, S> col_ind_;
  gt::gtensor<int, 1, S> row_ptr_;
};

} // namespace sparse

} // namespace gt

#endif
