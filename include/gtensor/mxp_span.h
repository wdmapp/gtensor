#ifndef GTENSOR_MXP_SPAN_H
#define GTENSOR_MXP_SPAN_H

#include "mxp_ambivalent.h"
#include <gtensor/gtensor.h>

// __________________________________________________________________________ //

namespace mxp
{

// -------------------------------------------------------------------------- //

template <typename T, gt::size_type N, typename S, typename X>
class mxp_span
  : public gt::gtensor_span<T, N, S>
  , public gt::expression<mxp_span<T, N, S, X>>
{
public:
  using self_type = mxp_span<T, N, S, X>;
  using base_type = gt::gtensor_span<T, N, S>;
  using expression_base_type = gt::expression<mxp_span<T, N, S, X>>;
  using value_type = typename mxp::detail::ambivalent_t<X, T>;

  using typename base_type::pointer;
  using typename base_type::shape_type;
  using typename base_type::strides_type;

  // ------------------------------------------------------------------------ //

  GT_INLINE mxp_span(pointer data, const shape_type& shape,
                     const strides_type& strides)
    : base_type(data, shape, strides)
  {}

  // ------------------------------------------------------------------------ //

  using base_type::operator=;

  // ------------------------------------------------------------------------ //

  using expression_base_type::derived;

  template <typename... Args>
  inline auto view(Args&&... args) const&
  {
    return gt::view(derived(), std::forward<Args>(args)...);
  }

  template <typename... Args>
  inline auto view(Args&&... args) &
  {
    return gt::view(derived(), std::forward<Args>(args)...);
  }

  template <typename... Args>
  inline auto view(Args&&... args) &&
  {
    return gt::view(std::move(*this).derived(), std::forward<Args>(args)...);
  }

  // ------------------------------------------------------------------------ //

  template <typename... Args>
  GT_INLINE auto operator()(Args&&... args) const -> value_type
  {
    return value_type(base_type::operator()(args...));
  }

  // ------------------------------------------------------------------------ //

  GT_INLINE auto data_access(gt::size_type i) const -> value_type
  {
    return value_type(base_type::data_access(i));
  }

  // ------------------------------------------------------------------------ //

  inline auto to_kernel() const -> self_type { return *this; }

  // ------------------------------------------------------------------------ //
};

// -------------------------------------------------------------------------- //

template <gt::size_type N, typename S, typename X, typename T>
GT_INLINE auto adapt(gt::space_pointer<T, S> data,
                     const gt::shape_type<N>& shape)
{
  return mxp_span<T, N, S, X>(data, shape, gt::calc_strides(shape));
}

// host
template <gt::size_type N, typename X, typename T>
GT_INLINE auto adapt(T* data, const gt::shape_type<N>& shape)
{
  return adapt<N, gt::space::host, X, T>(data, shape);
}

// device
template <gt::size_type N, typename X, typename T>
GT_INLINE auto adapt_device(T* data, const gt::shape_type<N>& shape)
{
  return adapt<N, gt::space::device, X, T>(gt::device_pointer_cast(data),
                                           shape);
}

// -------------------------------------------------------------------------- //

}; // namespace mxp

// __________________________________________________________________________ //

#endif // GTENSOR_MXP_SPAN_H
