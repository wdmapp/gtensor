
#ifndef GTENSOR_EXPRESSION_H
#define GTENSOR_EXPRESSION_H

#include "defs.h"
#include "gtensor_forward.h"
#include "gtl.h"
#include "helper.h"
#include "space.h"

namespace gt
{

template <typename T, size_type N, typename S>
struct gtensor_span;

// ======================================================================
// expression
//
// CRTP base class for all expressions

template <typename D>
class expression
{
protected:
  // cannot be directly instantiated
  expression() = default;

public:
  using derived_type = D;

  GT_INLINE const derived_type& derived() const&;
  GT_INLINE derived_type& derived() &;
  derived_type derived() &&;
};

template <typename D>
GT_INLINE auto expression<D>::derived() const& -> const derived_type&
{
  return static_cast<const derived_type&>(*this);
}

template <typename D>
GT_INLINE auto expression<D>::derived() & -> derived_type&
{
  return static_cast<derived_type&>(*this);
}

template <typename D>
inline auto expression<D>::derived() && -> derived_type
{
  return static_cast<derived_type&&>(*this);
}

// ======================================================================
// is_expression

template <class E>
using is_expression =
  std::is_base_of<expression<std::decay_t<E>>, std::decay_t<E>>;

// ======================================================================
// has_expression

template <typename... Args>
using has_expression = disjunction<is_expression<Args>...>;

// ======================================================================
// to_kernel_t

// Note: we want to preserve const, so don't use decay_t. Required for
// kernel view constness behavior to work correctly.
template <typename EC>
using to_kernel_t =
  decltype(std::declval<std::remove_reference_t<EC>>().to_kernel());

// ======================================================================
// index expression helper

template <typename E>
GT_INLINE decltype(auto) index_expression(E expr, shape_type<1> idx)
{
  return expr(idx[0]);
}

template <typename E>
GT_INLINE decltype(auto) index_expression(E expr, shape_type<2> idx)
{
  return expr(idx[0], idx[1]);
}

template <typename E>
GT_INLINE decltype(auto) index_expression(E expr, shape_type<3> idx)
{
  return expr(idx[0], idx[1], idx[2]);
}

template <typename E>
GT_INLINE decltype(auto) index_expression(E expr, shape_type<4> idx)
{
  return expr(idx[0], idx[1], idx[2], idx[3]);
}

template <typename E>
GT_INLINE decltype(auto) index_expression(E expr, shape_type<5> idx)
{
  return expr(idx[0], idx[1], idx[2], idx[3], idx[4]);
}

template <typename E>
GT_INLINE decltype(auto) index_expression(E expr, shape_type<6> idx)
{
  return expr(idx[0], idx[1], idx[2], idx[3], idx[4], idx[5]);
}

} // namespace gt

#endif
