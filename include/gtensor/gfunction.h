
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

template <typename S, typename F, typename... Es>
inline void calc_shape(S& shape, const F& e, const Es&... es)
{
  broadcast_shape(shape, e.shape());
  calc_shape(shape, es...);
}

template <typename S, typename TPL, size_type... I>
inline void calc_shape_expand(S& shape, const TPL& e, std::index_sequence<I...>)
{
  return calc_shape(shape, std::get<I>(e)...);
}

} // namespace detail

template <typename S, typename... E>
inline void calc_shape(S& shape, const std::tuple<E...>& e)
{
  for (auto& val : shape) {
    val = 1;
  }
  return detail::calc_shape_expand(shape, e,
                                   std::make_index_sequence<sizeof...(E)>());
}

// ======================================================================
// gfunction

template <typename F, typename... E>
class gfunction;

template <typename F, typename... E>
struct gtensor_inner_types<gfunction<F, E...>>
{
  using space_type = space_t<expr_space_type<E>...>;
  constexpr static size_type dimension = helper::calc_dimension<E...>();

  using value_type =
    decltype(std::declval<F>()(std::declval<expr_value_type<E>>()...));
  using reference = value_type;
  using const_reference = value_type;
};

template <typename F, typename... E>
class gfunction : public expression<gfunction<F, E...>>
{
public:
  using self_type = gfunction<F, E...>;
  using base_type = expression<self_type>;
  using inner_types = gtensor_inner_types<self_type>;
  using space_type = typename inner_types::space_type;
  using value_type = typename inner_types::value_type;
  using reference = typename inner_types::reference;
  using const_reference = typename inner_types::const_reference;

  constexpr static size_type dimension() { return inner_types::dimension; };

  using shape_type = gt::shape_type<dimension()>;

  gfunction(F&& f, E&&... e) : f_(std::forward<F>(f)), e_(std::forward<E>(e)...)
  {}

  shape_type shape() const;
  int shape(int i) const;

  template <typename... Args>
  GT_INLINE value_type operator()(Args... args) const;

  gfunction<F, to_kernel_t<E>...> to_kernel() const;

private:
  template <std::size_t... I, typename... Args>
  GT_INLINE value_type access(std::index_sequence<I...>, Args... args) const;

private:
  F f_;
  std::tuple<E...> e_;
};

// ----------------------------------------------------------------------
// gfunction implementation

template <typename F, typename... E>
inline auto gfunction<F, E...>::shape() const -> shape_type
{
  shape_type shape;
  calc_shape(shape, e_);
  return shape;
}

template <typename F, typename... E>
inline int gfunction<F, E...>::shape(int i) const
{
  return shape()[i];
}

template <typename F, typename... E>
template <typename... Args>
inline auto gfunction<F, E...>::operator()(Args... args) const -> value_type
{
  return access(std::make_index_sequence<sizeof...(E)>(), args...);
}

#pragma nv_exec_check_disable
template <typename F, typename... E>
template <std::size_t... I, typename... Args>
inline auto gfunction<F, E...>::access(std::index_sequence<I...>,
                                       Args... args) const -> value_type
{
  return f_(std::get<I>(e_)(args...)...);
}

template <typename F, typename... E>
auto function(F&& f, E&&... e)
{
  return gfunction<F, to_expression_t<E>...>(std::forward<F>(f),
                                             std::forward<E>(e)...);
}

namespace detail
{

template <typename F, std::size_t... I, typename... E>
inline auto make_gfunction(const F& f, std::index_sequence<I...>,
                           const std::tuple<E...>& e)
{
  return function(F(f), (std::get<I>(e).to_kernel())...);
}

} // namespace detail

template <typename F, typename... E>
inline gfunction<F, to_kernel_t<E>...> gfunction<F, E...>::to_kernel() const
{
  return detail::make_gfunction(f_, std::make_index_sequence<sizeof...(E)>(),
                                e_);
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

// ----------------------------------------------------------------------
// operator==, !-
//
// immediate evaluation
// FIXME, shoudl be done one device, too...

namespace detail
{

template <size_type N1, size_type N2>
struct equals
{
  template <typename E1, typename E2>
  static bool run(const E1& e1, const E2& e2)
  {
    return false;
  }
};

template <size_type N>
struct equals<N, N>
{
  static_assert(N != N, "comparison not yet implemented for this dimension");
};

template <>
struct equals<1, 1>
{
  template <typename E1, typename E2>
  static bool run(const E1& e1, const E2& e2)
  {
    if (e1.shape() != e2.shape()) {
      return false;
    }
    for (int i = 0; i < e1.shape(0); i++) {
      if (e1(i) != e2(i)) {
        return false;
      }
    }
    return true;
  }
};

template <>
struct equals<2, 2>
{
  template <typename E1, typename E2>
  static bool run(const E1& e1, const E2& e2)
  {
    if (e1.shape() != e2.shape()) {
      return false;
    }
    for (int j = 0; j < e1.shape(1); j++) {
      for (int i = 0; i < e1.shape(0); i++) {
        if (e1(i, j) != e2(i, j)) {
          return false;
        }
      }
    }
    return true;
  }
};

template <>
struct equals<3, 3>
{
  template <typename E1, typename E2>
  static bool run(const E1& e1, const E2& e2)
  {
    if (e1.shape() != e2.shape()) {
      return false;
    }
    for (int k = 0; k < e1.shape(2); k++) {
      for (int j = 0; j < e1.shape(1); j++) {
        for (int i = 0; i < e1.shape(0); i++) {
          if (e1(i, j, k) != e2(i, j, k)) {
            return false;
          }
        }
      }
    }
    return true;
  }
};

template <>
struct equals<4, 4>
{
  template <typename E1, typename E2>
  static bool run(const E1& e1, const E2& e2)
  {
    if (e1.shape() != e2.shape()) {
      return false;
    }

    for (int x = 0; x < e1.shape(3); x++) {
      for (int k = 0; k < e1.shape(2); k++) {
        for (int j = 0; j < e1.shape(1); j++) {
          for (int i = 0; i < e1.shape(0); i++) {
            if (e1(i, j, k, x) != e2(i, j, k, x)) {
              return false;
            }
          }
        }
      }
    }
    return true;
  }
};

template <>
struct equals<5, 5>
{
  template <typename E1, typename E2>
  static bool run(const E1& e1, const E2& e2)
  {
    if (e1.shape() != e2.shape()) {
      return false;
    }

    for (int y = 0; y < e1.shape(4); y++) {
      for (int x = 0; x < e1.shape(3); x++) {
        for (int k = 0; k < e1.shape(2); k++) {
          for (int j = 0; j < e1.shape(1); j++) {
            for (int i = 0; i < e1.shape(0); i++) {
              if (e1(i, j, k, x, y) != e2(i, j, k, x, y)) {
                return false;
              }
            }
          }
        }
      }
    }
    return true;
  }
};

template <>
struct equals<6, 6>
{
  template <typename E1, typename E2>
  static bool run(const E1& e1, const E2& e2)
  {
    if (e1.shape() != e2.shape()) {
      return false;
    }

    for (int z = 0; z < e1.shape(5); z++) {
      for (int y = 0; y < e1.shape(4); y++) {
        for (int x = 0; x < e1.shape(3); x++) {
          for (int k = 0; k < e1.shape(2); k++) {
            for (int j = 0; j < e1.shape(1); j++) {
              for (int i = 0; i < e1.shape(0); i++) {
                if (e1(i, j, k, x, y, z) != e2(i, j, k, x, y, z)) {
                  return false;
                }
              }
            }
          }
        }
      }
    }
    return true;
  }
};

} // namespace detail

template <typename E1, typename E2>
bool operator==(const expression<E1>& e1, const expression<E2>& e2)
{
  return detail::equals<E1::dimension(), E2::dimension()>::run(e1.derived(),
                                                               e2.derived());
}

template <typename E1, typename E2>
bool operator!=(const expression<E1>& e1, const expression<E2>& e2)
{
  return !(e1 == e2);
}

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
