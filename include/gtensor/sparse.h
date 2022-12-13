#ifndef GTENSOR_SPARSE_H
#define GTENSOR_SPARSE_H

#include <type_traits>

#include "gtensor.h"
#include "reductions.h"

#include <iostream>

namespace gt
{

namespace sparse
{

namespace detail
{

template <typename Tin>
struct UnaryOpNonzero
{
  GT_INLINE int operator()(Tin a) const { return a == Tin(0) ? 0 : 1; }
};

template <typename DataArray, typename S>
struct offsets_helper;

template <typename DataArray>
struct offsets_helper<DataArray, gt::space::host>
{
  static gt::gtensor<int, 1> run(DataArray& d_a_batches, int nbatches)
  {
    using T = typename DataArray::value_type;
    auto matrix_shape = gt::shape(d_a_batches.shape(0), d_a_batches.shape(1));
    int matrix_size = calc_size(matrix_shape);
    gt::gtensor<int, 1> nnz_offsets(gt::shape(nbatches + 1));
    nnz_offsets(0) = 0;
    for (int i = 1; i < nnz_offsets.shape(0); i++) {
      // hack to workaround redutions not accepting views. This works because
      // this particular view IS contiguous, so wrapping it in the span object
      // will cause the reduction arg to match template deduction and it will
      // produce the correct result. TODO: make this less ugly!
      auto batch_span = gt::adapt<2, T>(
        d_a_batches.data() + (i - 1) * matrix_size, matrix_shape);
      nnz_offsets(i) = nnz_offsets(i - 1) +
                       gt::transform_reduce(batch_span, 0, std::plus<int>{},
                                            UnaryOpNonzero<T>{});
    }
    return nnz_offsets;
  }
};

#ifdef GTENSOR_HAVE_DEVICE

template <typename DataArray>
struct offsets_helper<DataArray, gt::space::device>
{
  static gt::gtensor<int, 1> run(DataArray& d_a_batches, int nbatches)
  {
    using T = typename DataArray::value_type;
    auto matrix_shape = gt::shape(d_a_batches.shape(0), d_a_batches.shape(1));
    int matrix_size = calc_size(matrix_shape);
    gt::gtensor<int, 1> nnz_offsets(gt::shape(nbatches + 1));
    nnz_offsets(0) = 0;
    for (int i = 1; i < nnz_offsets.shape(0); i++) {
      // hack to workaround redutions not accepting views. This works because
      // this particular view IS contiguous, so wrapping it in the span object
      // will cause the reduction arg to match template deduction and it will
      // produce the correct result. TODO: make this less ugly!
      auto batch_span = gt::adapt_device<2, T>(
        gt::raw_pointer_cast(d_a_batches.data()) + (i - 1) * matrix_size,
        matrix_shape);
      nnz_offsets(i) = nnz_offsets(i - 1) +
                       gt::transform_reduce(batch_span, 0, std::plus<int>{},
                                            UnaryOpNonzero<T>{});
    }
    return nnz_offsets;
  }
};

#endif

template <typename DataArray>
gt::gtensor<int, 1> batch_nnz_offsets(DataArray& d_a_batches, int nbatches)
{
  using S = typename DataArray::space_type;
  return offsets_helper<DataArray, S>::run(d_a_batches, nbatches);
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
    auto h_nnz_offsets = detail::batch_nnz_offsets(d_a, 1);
    shape_ = d_a.shape();
    nnz_ = h_nnz_offsets(1);

    values_.resize({nnz_});
    col_ind_.resize({nnz_});
    row_ptr_.resize({shape_[0] + 1});

    gt::gtensor<int, 1, S> d_nnz_offsets{h_nnz_offsets.shape()};
    gt::copy(h_nnz_offsets, d_nnz_offsets);
    gt::synchronize();

    auto d_batches_view = d_a.view(gt::all, gt::all, gt::newaxis);
    convert_batches(d_batches_view, d_nnz_offsets);
  }

  template <typename BatchData>
  static auto join_matrix_batches(BatchData& d_matrix_batches)
  {
    static_assert(expr_dimension<BatchData>() == 3,
                  "batched sparse construction requires a 3d object");
    int nrows = d_matrix_batches.shape(0);
    int ncols = d_matrix_batches.shape(1);
    int nbatches = d_matrix_batches.shape(2);
    auto h_nnz_offsets = detail::batch_nnz_offsets(d_matrix_batches, nbatches);
    gt::gtensor<int, 1, S> d_nnz_offsets{h_nnz_offsets.shape()};
    csr_matrix csr_mat(gt::shape(nrows * nbatches, ncols * nbatches),
                       h_nnz_offsets(nbatches));

    gt::copy(h_nnz_offsets, d_nnz_offsets);

    csr_mat.convert_batches(d_matrix_batches, d_nnz_offsets);
    return csr_mat;
  }

  template <typename BatchView>
  void convert_batches(BatchView& d_matrix_view,
                       gt::gtensor<int, 1, S>& d_nnz_offsets)
  {
    static_assert(expr_dimension<BatchView>() == 3,
                  "3d view required for common dense conversion helper");
    auto k_nnz_offsets = d_nnz_offsets.to_kernel();
    auto k_matrix_view = d_matrix_view.to_kernel();
    int nrows = d_matrix_view.shape(0);
    int ncols = d_matrix_view.shape(1);
    int nbatches = d_matrix_view.shape(2);

    auto k_values_ = values_.to_kernel();
    auto k_col_ind_ = col_ind_.to_kernel();
    auto k_row_ptr_ = row_ptr_.to_kernel();
    int k_nnz_ = nnz_;

    // past all matrices along diagonal of sparse matrix
    gt::launch<1, S>(
      gt::shape(nbatches), GT_LAMBDA(int b) {
        int value_offset = k_nnz_offsets(b);
        int col_offset = ncols * b;
        int row_ptr_idx = nrows * b;
        T temp;
        // Note: we are doing a transpose, since CSR is row major
        for (int i = 0; i < nrows; i++) {
          bool row_first = true;
          for (int j = 0; j < ncols; j++) {
            temp = k_matrix_view(i, j, b);
            if (temp != T(0)) {
              k_values_(value_offset) = temp;
              k_col_ind_(value_offset) = col_offset + j;
              if (row_first) {
                k_row_ptr_(row_ptr_idx) = value_offset;
                row_first = false;
              }
              value_offset++;
            }
          }
          if (row_first) { // empty row
            if (b == 0 && i == 0) {
              k_row_ptr_(row_ptr_idx) = 0;
            } else {
              k_row_ptr_(row_ptr_idx) = k_row_ptr_(row_ptr_idx - 1);
            }
          }
          row_ptr_idx++;
        }
        if (b == nbatches - 1) {
          k_row_ptr_(k_row_ptr_.shape(0) - 1) = k_nnz_;
        }
      });
  }

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
