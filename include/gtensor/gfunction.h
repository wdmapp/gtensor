
#ifndef GTENSOR_GFUNCTION_H
#define GTENSOR_GFUNCTION_H

#include "defs.h"
#include "expression.h"
#include "gscalar.h"
#include "gstrided.h"
#include "helper.h"

#include <numeric>

namespace gt
{

// ======================================================================
// member_t

namespace detail
{

template <typename T, typename Enable = void>
struct member_type
{
  using type = gscalar<T>;
};

template <typename E>
struct member_type<E, std::enable_if_t<is_expression<std::decay_t<E>>::value>>
{
  using type = E;
};

} // namespace detail

template <typename E>
using to_expression_t = typename detail::member_type<E>::type;

// ======================================================================
// space_t

namespace detail
{

template <typename S, typename T>
struct space_type_op
{
  static_assert(!std::is_same<S, S>::value, "can't mix space types");
};

template <typename S>
struct space_type_op<S, S>
{
  using type = S;
};

template <>
struct space_type_op<space::any, space::any>
{
  using type = space::any;
};

template <typename S>
struct space_type_op<space::any, S>
{
  using type = S;
};

template <typename S>
struct space_type_op<S, space::any>
{
  using type = S;
};

template <typename...>
struct space_type;

template <typename S>
struct space_type<S>
{
  using type = S;
};

template <typename S, typename... R>
struct space_type<S, R...>
{
  using type = typename space_type_op<S, typename space_type<R...>::type>::type;
};

} // namespace detail

template <typename... S>
using space_t = typename detail::space_type<S...>::type;

// ======================================================================
// broadcast_shape

template <typename S, typename S2>
inline void broadcast_shape(S& to, const S2& from)
{
  // FIXME, rev it
  auto end_from = from.size() - 1;
  auto end_to = to.size() - 1;
  for (int i = 0; i < from.size(); i++) {
    if (from[end_from - i] == to[end_to - i]) {
      // nothing to do
    } else if (to[end_to - i] == 1) {
      // broadcasting to
      to[end_to - i] = from[end_from - i];
    } else if (from[end_from - i] == 1) {
      // broadcasting, from, nothing to do
    } else {
      throw std::runtime_error("cannot broadcast to = " + to_string(to) +
                               " from = " + to_string(from) + "\n");
    }
  }
}

// ======================================================================
// calc_shape

namespace detail
{

template <typename S, typename E>
inline void calc_shape(S& shape, const E& e)
{
  broadcast_shape(shape, e.shape());
}

template <typename S, typename E, typename... Es>
inline void calc_shape(S& shape, const E& e, const Es&... es)
{
  broadcast_shape(shape, e.shape());
  calc_shape(shape, es...);
}

} // namespace detail

template <typename S, typename... Es>
inline void calc_shape(S& shape, const Es&... es)
{
  for (auto& val : shape) {
    val = 1;
  }
  detail::calc_shape(shape, es...);
}

// ======================================================================
// gfunction

// dummy type used as second expression in gfunction for unary operators
// Note that this should not exist in a shared namespace, because of
// issues caused by ADL. It should not be used in code outside of gtensor,
// but it is 'exposed' in the sense that it will show up in type signatures.
struct gt_empty_expr
{};

// declare generic gfunction; unary and binary versions are defined below
template <typename F, typename E1, typename E2>
class gfunction;

template <typename F, typename E1, typename E2>
struct gtensor_inner_types<gfunction<F, E1, E2>>
{
  using space_type = space_t<expr_space_type<E1>, expr_space_type<E2>>;
  constexpr static size_type dimension = helper::calc_dimension<E1, E2>();

  using value_type = decltype(std::declval<F>()(
    std::declval<expr_value_type<E1>>(), std::declval<expr_value_type<E2>>()));
  using reference = value_type;
  using const_reference = value_type;
};

template <typename F, typename E>
struct gtensor_inner_types<gfunction<F, E, gt_empty_expr>>
{
  using space_type = space_t<expr_space_type<E>>;
  constexpr static size_type dimension = helper::calc_dimension<E>();

  using value_type =
    decltype(std::declval<F>()(std::declval<expr_value_type<E>>()));
  using reference = value_type;
  using const_reference = value_type;
};

template <typename F, typename E>
class gfunction<F, E, gt_empty_expr>
  : public expression<gfunction<F, E, gt_empty_expr>>
{
public:
  using self_type = gfunction<F, E, gt_empty_expr>;
  using base_type = expression<self_type>;
  using inner_types = gtensor_inner_types<self_type>;
  using space_type = typename inner_types::space_type;
  using value_type = typename inner_types::value_type;
  using reference = typename inner_types::reference;
  using const_reference = typename inner_types::const_reference;

  // Note: important for const correctness. See gview for explanation.
  using const_kernel_type =
    gfunction<F, to_kernel_t<std::add_const_t<E>>, gt_empty_expr>;
  using kernel_type = const_kernel_type;

  constexpr static size_type dimension() { return inner_types::dimension; };

  using shape_type = gt::shape_type<dimension()>;

  gfunction(F&& f, E&& e) : f_(std::forward<F>(f)), e_(std::forward<E>(e)) {}

  shape_type shape() const;
  int shape(int i) const;
  GT_INLINE size_type size() const { return shape().size(); };

  template <typename... Args>
  GT_INLINE value_type operator()(Args... args) const;

  const_kernel_type to_kernel() const;

private:
  F f_;
  E e_;
};

template <typename F, typename E1, typename E2>
class gfunction : public expression<gfunction<F, E1, E2>>
{
public:
  using self_type = gfunction<F, E1, E2>;
  using base_type = expression<self_type>;
  using inner_types = gtensor_inner_types<self_type>;
  using space_type = typename inner_types::space_type;
  using value_type = typename inner_types::value_type;
  using reference = typename inner_types::reference;
  using const_reference = typename inner_types::const_reference;

  // Note: important for const correctness. See gview for explanation.
  using const_kernel_type = gfunction<F, to_kernel_t<std::add_const_t<E1>>,
                                      to_kernel_t<std::add_const_t<E2>>>;
  using kernel_type = const_kernel_type;

  constexpr static size_type dimension() { return inner_types::dimension; };

  using shape_type = gt::shape_type<dimension()>;

  gfunction(F&& f, E1&& e1, E2&& e2)
    : f_(std::forward<F>(f)),
      e1_(std::forward<E1>(e1)),
      e2_(std::forward<E2>(e2))
  {}

  shape_type shape() const;
  int shape(int i) const;

  template <typename... Args>
  GT_INLINE value_type operator()(Args... args) const;

  const_kernel_type to_kernel() const;

private:
  F f_;
  E1 e1_;
  E2 e2_;
};

// ----------------------------------------------------------------------
// gfunction implementation

template <typename F, typename E>
inline auto gfunction<F, E, gt_empty_expr>::shape() const -> shape_type
{
  shape_type shape;
  calc_shape(shape, e_);
  return shape;
}

template <typename F, typename E>
inline int gfunction<F, E, gt_empty_expr>::shape(int i) const
{
  return shape()[i];
}

template <typename F, typename E1, typename E2>
inline auto gfunction<F, E1, E2>::shape() const -> shape_type
{
  shape_type shape;
  calc_shape(shape, e1_, e2_);
  return shape;
}

template <typename F, typename E1, typename E2>
inline int gfunction<F, E1, E2>::shape(int i) const
{
  return shape()[i];
}

template <typename F, typename E>
template <typename... Args>
GT_INLINE auto gfunction<F, E, gt_empty_expr>::operator()(Args... args) const
  -> value_type
{
  return f_(e_(args...));
}

template <typename F, typename E1, typename E2>
template <typename... Args>
GT_INLINE auto gfunction<F, E1, E2>::operator()(Args... args) const
  -> value_type
{
  return f_(e1_(args...), e2_(args...));
}

template <typename F, typename E>
auto function(F&& f, E&& e)
{
  return gfunction<F, to_expression_t<E>, gt_empty_expr>(std::forward<F>(f),
                                                         std::forward<E>(e));
}

template <typename F, typename E1, typename E2>
auto function(F&& f, E1&& e1, E2&& e2)
{
  return gfunction<F, to_expression_t<E1>, to_expression_t<E2>>(
    std::forward<F>(f), std::forward<E1>(e1), std::forward<E2>(e2));
}

template <typename F, typename E>
inline auto gfunction<F, E, gt_empty_expr>::to_kernel() const
  -> const_kernel_type
{
  return function(F(f_), e_.to_kernel());
}

template <typename F, typename E1, typename E2>
inline auto gfunction<F, E1, E2>::to_kernel() const -> const_kernel_type
{
  return function(F(f_), e1_.to_kernel(), e2_.to_kernel());
}

#define MAKE_UNARY_OP(NAME, OP)                                                \
  namespace ops                                                                \
  {                                                                            \
                                                                               \
  struct NAME                                                                  \
  {                                                                            \
    template <typename T>                                                      \
    GT_INLINE auto operator()(T a) const                                       \
    {                                                                          \
      return OP a;                                                             \
    }                                                                          \
  };                                                                           \
                                                                               \
  } /* namespace ops */                                                        \
                                                                               \
  template <typename E,                                                        \
            typename Enable = std::enable_if_t<has_expression<E>::value>>      \
  auto operator OP(E&& e)                                                      \
  {                                                                            \
    return function(ops::NAME{}, std::forward<E>(e));                          \
  }

MAKE_UNARY_OP(negate, -)

#undef MAKE_UNARY_OP

#define MAKE_BINARY_OP(NAME, OP)                                               \
  namespace ops                                                                \
  {                                                                            \
                                                                               \
  struct NAME                                                                  \
  {                                                                            \
    template <typename T, typename U>                                          \
    GT_INLINE auto operator()(T a, U b) const                                  \
    {                                                                          \
      return a OP b;                                                           \
    }                                                                          \
  };                                                                           \
                                                                               \
  } /* namespace ops */                                                        \
                                                                               \
  template <typename E1, typename E2,                                          \
            typename Enable = std::enable_if_t<has_expression<E1, E2>::value>> \
  auto operator OP(E1&& e1, E2&& e2)                                           \
  {                                                                            \
    return function(ops::NAME{}, std::forward<E1>(e1), std::forward<E2>(e2));  \
  }

MAKE_BINARY_OP(plus, +)
MAKE_BINARY_OP(minus, -)
MAKE_BINARY_OP(multiply, *)
MAKE_BINARY_OP(divide, /)

#undef MAKE_BINARY_OP

#define MAKE_UNARY_FUNC(NAME, FUNC)                                            \
                                                                               \
  namespace funcs                                                              \
  {                                                                            \
  struct NAME                                                                  \
  {                                                                            \
    template <typename T>                                                      \
    GT_INLINE auto operator()(T a) const                                       \
    {                                                                          \
      return FUNC(a);                                                          \
    }                                                                          \
  };                                                                           \
  }                                                                            \
                                                                               \
  template <typename E,                                                        \
            typename Enable = std::enable_if_t<has_expression<E>::value>>      \
  auto NAME(E&& e)                                                             \
  {                                                                            \
    return function(funcs::NAME{}, std::forward<E>(e));                        \
  }

MAKE_UNARY_FUNC(abs, std::abs)
MAKE_UNARY_FUNC(sin, std::sin)
MAKE_UNARY_FUNC(cos, std::cos)
MAKE_UNARY_FUNC(tan, std::tan)
MAKE_UNARY_FUNC(exp, std::exp)

#undef MAKE_UNARY_FUNC

// ======================================================================
// ggenerator

template <int N, typename T, typename F> // FIXME, T could be derived
class ggenerator : public expression<ggenerator<N, T, F>>
{
public:
  using value_type = T;
  using space_type = space::any;
  using shape_type = gt::shape_type<N>;
  constexpr static size_type dimension() { return N; }

  ggenerator(const shape_type& shape, const F& f) : shape_(shape), f_(f) {}

  shape_type shape() const { return shape_; }
  int shape(int i) const { return shape_[i]; }
  size_type size() const { return calc_size(shape()); }

  template <typename... Args>
  GT_INLINE auto operator()(Args... args) const
  {
    return f_(args...);
  }

  ggenerator to_kernel() const { return *this; }

private:
  shape_type shape_;
  F f_;
};

template <int N, typename T, typename F>
auto generator(const gt::shape_type<N>& shape, F&& f)
{
  return ggenerator<N, T, F>(shape, std::forward<F>(f));
  ;
}

} // namespace gt

#endif
