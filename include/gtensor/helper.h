
#ifndef GTENSOR_HELPER_H
#define GTENSOR_HELPER_H

#include "defs.h"
#include "meta.h"
#include "sarray.h"

#include <tuple>

namespace gt
{

template <typename... Args>
GT_INLINE auto shape(Args... args)
{
  return shape_type<sizeof...(args)>(std::forward<Args>(args)...);
}

// ======================================================================
// expr_value_type

template <typename E>
using expr_value_type = typename std::decay_t<E>::value_type;

// ======================================================================
// expr_reference_type

template <typename E>
using expr_reference_type = typename std::decay_t<E>::reference;

// ======================================================================
// expr_space_type

template <typename E>
using expr_space_type = typename std::decay_t<E>::space_type;

// ======================================================================
// expr_storage_type

template <typename E>
using expr_storage_type = typename std::decay_t<E>::storage_type;

// ======================================================================
// expr_shape_type

template <typename E>
using expr_shape_type = typename std::decay_t<E>::shape_type;

// ======================================================================
// expr_dimension

template <typename E>
constexpr size_type expr_dimension()
{
  return std::decay_t<E>::dimension();
}

// ======================================================================
// has_data_method

template <typename T, typename = void>
struct has_data_method : std::false_type
{};

template <typename T>
struct has_data_method<T, gt::meta::void_t<decltype(std::declval<T>().data())>>
  : std::true_type
{};

template <typename T>
constexpr bool has_data_method_v = has_data_method<T>::value;

// ======================================================================
// has_strides_method

template <typename T, typename = void>
struct has_strides_method : std::false_type
{};

template <typename T>
struct has_strides_method<
  T, gt::meta::void_t<decltype(std::declval<T>().strides())>> : std::true_type
{};

template <typename T>
constexpr bool has_strides_method_v = has_strides_method<T>::value;

// ======================================================================
// has_size_method

template <typename T, typename = void>
struct has_size_method : std::false_type
{};

template <typename T>
struct has_size_method<T, gt::meta::void_t<decltype(std::declval<T>().size())>>
  : std::true_type
{};

template <typename T>
constexpr bool has_size_method_v = has_size_method<T>::value;

// ======================================================================
// has_container_methods

template <typename T, typename = void>
struct has_container_methods : std::false_type
{};

template <typename T>
struct has_container_methods<
  T, gt::meta::void_t<
       std::enable_if_t<has_data_method_v<T> && has_size_method_v<T>>>>
  : std::true_type
{};

template <typename T>
constexpr bool has_container_methods_v = has_container_methods<T>::value;

namespace helper
{

// ======================================================================
// nd_initializer_list_t

namespace detail
{

template <typename T, size_type N>
struct nd_initializer_list
{
  using type =
    std::initializer_list<typename nd_initializer_list<T, N - 1>::type>;
};

template <typename T>
struct nd_initializer_list<T, 1>
{
  using type = std::initializer_list<T>;
};

} // namespace detail

template <typename T, size_type N>
using nd_initializer_list_t = typename detail::nd_initializer_list<T, N>::type;

// ======================================================================
// nd_initializer_list_shape

namespace detail
{

template <size_type N, typename IL>
struct nd_init_list_shape;

template <typename IL>
struct nd_init_list_shape<1, IL>
{
  constexpr static shape_type<1> run(IL il) { return {int(il.size())}; }
};

template <typename IL>
struct nd_init_list_shape<2, IL>
{
  constexpr static shape_type<2> run(IL il)
  {
    return {int(il.size() == 0 ? 0 : il.begin()->size()), int(il.size())};
  }
};

template <typename IL>
struct nd_init_list_shape<3, IL>
{
  constexpr static shape_type<3> run(IL il)
  {
    return {int((il.size() == 0 || il.begin()->size() == 0)
                  ? 0
                  : il.begin()->begin()->size()),
            int(il.size() == 0 ? 0 : il.begin()->size()), int(il.size())};
  }
};

} // namespace detail

template <size_type N, typename IL>
constexpr shape_type<N> nd_initializer_list_shape(IL il)
{
  return detail::nd_init_list_shape<N, IL>::run(il);
}

// ======================================================================
// nd_initializer_list_copy

namespace detail
{

template <size_type N, typename IL, typename T>
struct nd_init_list_copy;

template <typename IL, typename T>
struct nd_init_list_copy<1, IL, T>
{
  constexpr static void run(IL il, T& t)
  {
    int i = 0;
    for (auto val : il) {
      t(i) = val;
      i++;
    }
  }
};

template <typename IL, typename T>
struct nd_init_list_copy<2, IL, T>
{
  constexpr static void run(IL il, T& t)
  {
    int j = 0;
    for (auto il0 : il) {
      int i = 0;
      for (auto val : il0) {
        t(i, j) = val;
        i++;
      }
      j++;
    }
  }
};

template <typename IL, typename T>
struct nd_init_list_copy<3, IL, T>
{
  constexpr static void run(IL il, T& t)
  {
    int k = 0;
    for (auto il0 : il) {
      int j = 0;
      for (auto il1 : il0) {
        int i = 0;
        for (auto val : il1) {
          t(i, j, k) = val;
          i++;
        }
        j++;
      }
      k++;
    }
  }
};

} // namespace detail

template <size_type N, typename IL, typename T>
constexpr void nd_initializer_list_copy(IL il, T& t)
{
  return detail::nd_init_list_copy<N, IL, T>::run(il, t);
}

// ======================================================================
// calc_dimension
//
// computes max dimension across expressions

namespace detail
{
template <typename... Es>
struct calc_dimension;

template <typename E>
struct calc_dimension<E>
{
  constexpr static int N = expr_dimension<E>();
};

template <typename F, typename... Rs>
struct calc_dimension<F, Rs...>
{
  constexpr static size_type dim0 = expr_dimension<F>();
  constexpr static size_type dim1 = calc_dimension<Rs...>::N;
  constexpr static int N = (dim0 > dim1) ? dim0 : dim1;
};

} // namespace detail

template <typename... Es>
inline constexpr size_type calc_dimension()
{
  return detail::calc_dimension<Es...>::N;
}

// ======================================================================
// max
//
// calculate max(f(e) for e in E...)

namespace detail
{
template <size_type I, typename F, typename... E>
inline std::enable_if_t<I == sizeof...(E), size_type> max(
  F&& f, const std::tuple<E...>& tpl)
{
  return 0;
}

template <size_type I, typename F, typename... E>
inline std::enable_if_t<I + 1 == sizeof...(E), size_type> max(
  F&& f, const std::tuple<E...>& tpl)
{
  size_type val = std::forward<F>(f)(std::get<I>(tpl));
  return val;
}

template <size_type I, typename F, typename... E>
inline std::enable_if_t<I + 1 < sizeof...(E), size_type> max(
  F&& f, const std::tuple<E...>& tpl)
{
  size_type val = std::forward<F>(f)(std::get<I>(tpl));
  return std::max(val, max<I + 1, F, E...>(std::forward<F>(f), tpl));
}

} // namespace detail

template <typename F, typename... E>
size_type max(F&& f, const std::tuple<E...>& tpl)
{
  return detail::max<0, F, E...>(std::forward<F>(f), tpl);
}

} // namespace helper

// ======================================================================
// is_allowed_element_type_conversion

template <typename From, typename To>
struct is_allowed_element_type_conversion
  : std::is_convertible<From (*)[], To (*)[]>
{};

} // namespace gt

#endif
