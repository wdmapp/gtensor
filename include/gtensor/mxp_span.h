#ifndef GTENSOR_MXP_SPAN_H
#define GTENSOR_MXP_SPAN_H

#include <gtensor/gtensor.h>
#include "mxp_ambivalent.h"

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

  template <typename... Args>
  GT_INLINE auto operator()(Args&&... args) const -> value_type
  {
    return value_type(base_type::operator()(args...));
  }

  // ------------------------------------------------------------------------ //
};

// -------------------------------------------------------------------------- //

template <gt::size_type N, typename X, typename T>
GT_INLINE auto adapt(T* data, const gt::shape_type<N>& shape)
{
  return mxp_span<T, N, gt::space::host, X>(data, shape,
                                            gt::calc_strides(shape));
}

// -------------------------------------------------------------------------- //

}; // namespace mxp

// __________________________________________________________________________ //

#endif // GTENSOR_MXP_SPAN_H
