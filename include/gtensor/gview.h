
#ifndef GTENSOR_GVIEW_H
#define GTENSOR_GVIEW_H

#include "assign.h"
#include "expression.h"
#include "gslice.h"
#include "meta.h"

namespace gt
{

// ----------------------------------------------------------------------
// gview_adaptor

namespace detail
{

template <typename E>
class gview_adaptor
{
public:
  using space_type = expr_space_type<E>;
  using shape_type = expr_shape_type<E>;

  using inner_expression_type = std::decay_t<E>;
  using value_type = typename inner_expression_type::value_type;
  using reference = typename inner_expression_type::reference;
  using const_reference = typename inner_expression_type::const_reference;

  gview_adaptor(E&& e) : e_(e), strides_(calc_strides(e.shape())) {}

  shape_type shape() const { return e_.shape(); }
  shape_type strides() const { return strides_; }

  decltype(auto) to_kernel() const { return e_.to_kernel(); }
  decltype(auto) to_kernel() { return e_.to_kernel(); }

  GT_INLINE size_type size() const { return e_.size(); };

  GT_INLINE decltype(auto) data_access(size_type i) const
  {
    shape_type idx = unravel(i, strides_);
    return access(std::make_index_sequence<idx.size()>(), idx);
  }

  GT_INLINE decltype(auto) data_access(size_type i)
  {
    shape_type idx = unravel(i, strides_);
    return access(std::make_index_sequence<idx.size()>(), idx);
  }

private:
  template <size_type... I>
  GT_INLINE decltype(auto) access(std::index_sequence<I...>,
                                  const shape_type& idx) const
  {
    return e_(idx[I]...);
  }

  template <size_type... I>
  GT_INLINE decltype(auto) access(std::index_sequence<I...>,
                                  const shape_type& idx)
  {
    return e_(idx[I]...);
  }

  E e_;
  shape_type strides_;
};

} // namespace detail

// ----------------------------------------------------------------------
// select_gview_adaptor

namespace detail
{

template <typename E, typename Enable = void>
struct select_gview_adaptor
{
  using type = detail::gview_adaptor<E>;
};

template <typename E>
struct select_gview_adaptor<
  E, gt::meta::void_t<decltype(std::declval<E>().strides())>>
{
  using type = E;
};

// special requirements for reshape, which replaces strides rather than
// adapting them
template <typename E, typename Enable = void>
struct select_reshape_gview_adaptor
{
  using type = detail::gview_adaptor<E>;
};

// gcontainers and gtensor_spans have the default stride and layout, so they
// are safe to override without layering via an adaptor
template <typename E>
struct select_reshape_gview_adaptor<
  E, gt::meta::void_t<
       std::enable_if_t<is_gcontainer<E>::value || is_gtensor_span<E>::value>>>
{
  using type = E;
};

} // namespace detail

template <typename E>
using select_gview_adaptor_t = typename detail::select_gview_adaptor<E>::type;

template <typename E>
using select_reshape_gview_adaptor_t =
  typename detail::select_reshape_gview_adaptor<E>::type;

// ======================================================================
// gview

template <typename EC, size_type N>
class gview;

template <typename EC, size_type N>
struct gtensor_inner_types<gview<EC, N>>
{
  using space_type = expr_space_type<EC>;
  constexpr static size_type dimension = N;

  // Note: we want to preserve const, so don't use decay_t. While currently
  // tests pass either way, this is useful for debugging and may become
  // an issue later.
  using inner_expression_type = std::remove_reference_t<EC>;
  using value_type = typename inner_expression_type::value_type;
  using reference = typename inner_expression_type::reference;
  using const_reference = typename inner_expression_type::const_reference;
};

template <typename EC, size_type N>
class gview : public gstrided<gview<EC, N>>
{
public:
  using self_type = gview<EC, N>;
  using base_type = gstrided<self_type>;
  using inner_types = gtensor_inner_types<self_type>;

  using const_reference = typename inner_types::const_reference;
  using reference = typename inner_types::reference;
  using value_type = typename inner_types::value_type;

  using typename base_type::shape_type;
  using typename base_type::strides_type;

  // Note: important for const correctness. If the gview is const and owns
  // a gtensor object, rather than holding a reference, then the const
  // version of methods of the underlying object will be called. The const
  // to_kernel method will return kernel const adapted kernel view objects,
  // even though the data member inside gview is not const (the type system
  //  and to_kernel_t macro don't see this without some help).
  using const_kernel_type = gview<to_kernel_t<std::add_const_t<EC>>, N>;
  using kernel_type = gview<to_kernel_t<EC>, N>;

  gview(EC&& e, size_type offset, const shape_type& shape,
        const strides_type& strides);
  gview(const gview&) = default;
  gview(gview&&) = default;
  self_type& operator=(self_type&& o) = default;

  self_type& operator=(const self_type& o);
  template <typename E>
  self_type& operator=(const expression<E>& e);
  self_type& operator=(value_type val);

  void fill(const value_type v);

  const_kernel_type to_kernel() const;
  kernel_type to_kernel();

  // Note: Returns either const_reference or reference. Uses decltype(auto) to
  // reflect the const-depth of the underyling expression type. In particular,
  // if containing a gtensor_span object or reference, the gview should also be
  // shallow const. If containing a gtensor object or reference, the gview
  // should be deep const.
  template <typename... Args>
  GT_INLINE decltype(auto) operator()(Args&&... args) const;
  template <typename... Args>
  GT_INLINE decltype(auto) operator()(Args&&... args);

  GT_INLINE decltype(auto) operator[](const shape_type& idx) const;
  GT_INLINE decltype(auto) operator[](const shape_type& idx);

  GT_INLINE decltype(auto) data_access(size_type i) const;
  GT_INLINE decltype(auto) data_access(size_type i);

private:
  EC e_;
  size_type offset_;

  friend class gstrided<self_type>;

  template <typename S, size_type... I>
  GT_INLINE decltype(auto) access(std::index_sequence<I...>, const S& idx) const
  {
    return (*this)(idx[I]...);
  }

  template <typename S, size_type... I>
  GT_INLINE decltype(auto) access(std::index_sequence<I...>, const S& idx)
  {
    return (*this)(idx[I]...);
  }
};

// ======================================================================
// gview implementation

template <typename EC, size_type N>
inline gview<EC, N>::gview(EC&& e, size_type offset, const shape_type& shape,
                           const strides_type& strides)
  : base_type(shape, strides), e_(std::forward<EC>(e)), offset_(offset)
{}

template <typename EC, size_type N>
inline auto gview<EC, N>::to_kernel() const -> const_kernel_type
{
  return const_kernel_type(e_.to_kernel(), offset_, this->shape(),
                           this->strides());
}

template <typename EC, size_type N>
inline auto gview<EC, N>::to_kernel() -> kernel_type
{
  return kernel_type(e_.to_kernel(), offset_, this->shape(), this->strides());
}

template <typename EC, size_type N>
template <typename... Args>
GT_INLINE decltype(auto) gview<EC, N>::operator()(Args&&... args) const
{
  return data_access(base_type::index(std::forward<Args>(args)...));
}

template <typename EC, size_type N>
template <typename... Args>
GT_INLINE decltype(auto) gview<EC, N>::operator()(Args&&... args)
{
  return data_access(base_type::index(std::forward<Args>(args)...));
}

template <typename EC, size_type N>
GT_INLINE decltype(auto) gview<EC, N>::operator[](const shape_type& idx) const
{
  return access(std::make_index_sequence<shape_type::dimension>(), idx);
}

template <typename EC, size_type N>
GT_INLINE decltype(auto) gview<EC, N>::operator[](const shape_type& idx)
{
  return access(std::make_index_sequence<shape_type::dimension>(), idx);
}

template <typename EC, size_type N>
GT_INLINE decltype(auto) gview<EC, N>::data_access(size_t i) const
{
  return e_.data_access(offset_ + i);
}

template <typename EC, size_type N>
GT_INLINE decltype(auto) gview<EC, N>::data_access(size_t i)
{
  return e_.data_access(offset_ + i);
}

template <typename EC, size_type N>
inline auto gview<EC, N>::operator=(const gview<EC, N>& o) -> gview&
{
  assign(*this, o);
  return *this;
}

template <typename EC, size_type N>
template <typename E>
inline auto gview<EC, N>::operator=(const expression<E>& e) -> gview&
{
  assign(*this, e.derived());
  return *this;
}

template <typename EC, size_type N>
inline auto gview<EC, N>::operator=(value_type val) -> gview&
{
  assign(*this, scalar(val));
  return *this;
}

template <typename EC, size_type N>
inline void gview<EC, N>::fill(const value_type v)
{
  assign(*this, scalar(v));
}

// ======================================================================
// view

template <size_type N, typename E>
auto view(E&& _e, const std::vector<gdesc>& descs)
{
  using EC = select_gview_adaptor_t<E>;

  EC e(std::forward<E>(_e));

  size_type offset = 0;
  const auto& old_shape = e.shape();
  const auto& old_strides = e.strides();
  gt::shape_type<N> shape, strides;
  int new_i = 0, old_i = 0;
  for (int i = 0; i < descs.size(); i++) {
    // std::cout << "slice " << descs[i] << " old_shape " << old_shape[i]
    //           << " old_stride " << old_strides[i] << "\n";
    if (descs[i].type() == gdesc::ALL) {
      shape[new_i] = old_shape[old_i];
      strides[new_i] = old_strides[old_i];
      new_i++;
      old_i++;
    } else if (descs[i].type() == gdesc::VALUE) {
      offset += descs[i].value() * old_strides[old_i];
      old_i++;
    } else if (descs[i].type() == gdesc::NEWAXIS) {
      shape[new_i] = 1;
      strides[new_i] = 0;
      new_i++;
    } else if (descs[i].type() == gdesc::SLICE) {
      auto slice = descs[i].slice();
      int start = slice.start;
      int stop = slice.stop;
      int step = slice.step;
      if (step == gslice::none) {
        step = 1;
      }
      if (step == 0) {
        throw std::runtime_error(
          "view: the step parameter in a slice cannot be zero!");
      }
      if (start == gslice::none) {
        start = step > 0 ? 0 : old_shape[old_i] - 1;
      } else if (start < 0) {
        start += old_shape[old_i];
      }
      if (stop == gslice::none) {
        stop = step > 0 ? old_shape[old_i] : -1;
      } else if (stop == 0 && step == 1) {
        // FIXME, keep this?, different from numpy, though convenient
        stop += old_shape[old_i];
      } else if (stop < 0) {
        stop += old_shape[old_i];
      }
      // std::cout << "slice " << start << ":" << stop << ":" << step << "\n";
      // FIXME? Could just return 0-size
      if (step > 0 && start >= stop) {
        throw std::runtime_error("view: start must be less than stop!");
      }
      if (step < 0 && stop >= start) {
        throw std::runtime_error("view: start must be greater than stop!");
      }
      if ((step > 0 && stop > old_shape[old_i]) ||
          (step < 0 && start > old_shape[old_i])) {
        throw std::runtime_error("view: cannot exceed underlying shape!");
      }
      if (step > 0) {
        shape[new_i] = (stop - start - 1) / step + 1;
      } else {
        shape[new_i] = (start - stop - 1) / (-step) + 1;
      }
      strides[new_i] = old_strides[old_i] * step;
      offset += start * old_strides[old_i];
      new_i++;
      old_i++;
    } else {
      assert(0);
    }
  }
  // handle rest as if filled with gt::all
  while (old_i < old_shape.size()) {
    shape[new_i] = old_shape[old_i];
    strides[new_i] = old_strides[old_i];
    new_i++;
    old_i++;
  }
  // for (int i = 0; i < new_i; i++) {
  //   std::cout << i << " new shape " << shape[i] << " stride " << strides[i]
  //   <<
  //   "\n";
  // }

  assert(new_i == N);
  return gview<EC, N>(std::forward<EC>(e), offset, shape, strides);
}

template <typename E, typename... Args>
auto view(E&& e, Args&&... args)
{
  constexpr std::size_t N = view_dimension<E, Args...>();
  std::vector<gdesc> descs{std::forward<Args>(args)...};
  return view<N>(std::forward<E>(e), descs);
}

// ======================================================================
// reshape

template <size_type N, typename E>
inline auto reshape(E&& _e, gt::shape_type<N> shape)
{
  // Note: use gview adapter more broadly, in particular for nested views,
  // since this routine simply replaces strides. The adapter is not needed
  // for containers and spans, because their data layout implicitly encodes
  // their striding. gviews on the other hand, do require an adapter.
  using EC = select_reshape_gview_adaptor_t<E>;

  EC e(std::forward<E>(_e));

  size_type size_e = e.size();
  size_type size = 1;
  int dim_adjust = -1;
  for (int d = 0; d < N; d++) {
    if (shape[d] == -1) {
      assert(dim_adjust == -1); // can at most have one placeholder
      dim_adjust = d;
    } else {
      size *= shape[d];
    }
  }
  if (dim_adjust == -1) {
    assert(size == e.size());
  } else {
    assert(e.size() % size == 0);
    shape[dim_adjust] = e.size() / size;
  }
  assert(calc_size(shape) == calc_size(e.shape()));
  return gview<EC, N>(std::forward<EC>(e), 0, shape, calc_strides(shape));
}

// ======================================================================
// flatten

template <typename E>
inline auto flatten(E&& e)
{
  return reshape(std::forward<E>(e), shape(e.size()));
}

// ======================================================================
// swapaxes

template <typename E>
inline auto swapaxes(E&& _e, int axis1, int axis2)
{
  using EC = select_gview_adaptor_t<E>;

  EC e(std::forward<E>(_e));

  constexpr int N = expr_dimension<E>();
  expr_shape_type<E> shape;
  expr_shape_type<E> strides;

  for (int d = 0; d < shape.size(); d++) {
    if (d == axis1) {
      shape[d] = e.shape()[axis2];
      strides[d] = e.strides()[axis2];
    } else if (d == axis2) {
      shape[d] = e.shape()[axis1];
      strides[d] = e.strides()[axis1];
    } else {
      shape[d] = e.shape()[d];
      strides[d] = e.strides()[d];
    }
  }
  // FIXME, afterwards it's not col-major order anymore, os unravel will go
  // wrong
  // FIXME, could use sanity checks
  return gview<EC, N>(std::forward<EC>(e), 0, shape, strides);
}

// ======================================================================
// transpose

template <typename E>
inline auto transpose(E&& _e, expr_shape_type<E> axes)
{
  using EC = select_gview_adaptor_t<E>;

  EC e(std::forward<E>(_e));

  constexpr int N = expr_dimension<E>();
  expr_shape_type<E> shape;
  expr_shape_type<E> strides;

  for (int d = 0; d < shape.size(); d++) {
    shape[d] = e.shape()[axes[d]];
    strides[d] = e.strides()[axes[d]];
  }
  // FIXME, could use sanity checks
  return gview<EC, N>(std::forward<E>(e), 0, shape, strides);
}

} // namespace gt

#endif
