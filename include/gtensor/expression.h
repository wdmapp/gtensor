
#ifndef GTENSOR_EXPRESSION_H
#define GTENSOR_EXPRESSION_H

#include "defs.h"
#include "gtl.h"
#include "helper.h"
#include "space.h"

namespace gt
{

// fwd decl FIXME?
template <typename T, int N, typename S>
struct gtensor;

template <typename T, int N, typename S>
struct gtensor_view;

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

  GT_INLINE const derived_type& derived() const;
  GT_INLINE derived_type& derived();
};

template <typename D>
inline auto expression<D>::derived() const -> const derived_type&
{
  return static_cast<const derived_type&>(*this);
}

template <typename D>
inline auto expression<D>::derived() -> derived_type&
{
  return static_cast<derived_type&>(*this);
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

template <typename EC>
using to_kernel_t = decltype(std::declval<std::decay_t<EC>>().to_kernel());

// ----------------------------------------------------------------------
// 1-d, 2-d ostream output

namespace detail
{

template <size_type N>
struct expression_printer
{
  static_assert(N != N, "output not support for this dimension");
};

template <>
struct expression_printer<1>
{
  template <typename E>
  static void print_to(std::ostream& os, const E& e)
  {
    os << "{";
    for (int i = 0; i < e.shape(0); i++) {
      os << " " << e(i);
    }
    os << " }";
  }
};

template <>
struct expression_printer<2>
{
  template <typename E>
  static void print_to(std::ostream& os, const E& e)
  {
    os << "{";
    for (int j = 0; j < e.shape(1); j++) {
      os << "{";
      for (int i = 0; i < e.shape(0); i++) {
        os << " " << e(i, j);
      }
      os << " }";
      if (j < e.shape(1) - 1) {
        os << "\n";
      }
    }
    os << "}";
  }
};

} // namespace detail

template <typename E,
          typename Enable = std::enable_if_t<is_expression<E>::value>>
inline std::ostream& operator<<(std::ostream& os, const E& e)
{
  detail::expression_printer<expr_dimension<E>()>::print_to(os, e);
  return os;
}

} // namespace gt

#endif
