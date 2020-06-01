#ifndef GTENSOR_OPERATOR_H
#define GTENSOR_OPERATOR_H

#include "defs.h"
#include "gtl.h"
#include "helper.h"
#include "space.h"

#include <iostream>

namespace gt
{

// ----------------------------------------------------------------------
// EnableIf helpers for expression spaces

namespace detail
{

template <typename E>
using EnableIfDevice =
  std::enable_if_t<std::is_same<expr_space_type<E>, space::device>::value>;

template <typename E>
using EnableIfHost =
  std::enable_if_t<std::is_same<expr_space_type<E>, space::host>::value>;

template <typename E1, typename E2>
using EnableIfDeviceDevice =
  std::enable_if_t<std::is_same<expr_space_type<E1>, space::device>::value &&
                   std::is_same<expr_space_type<E1>, space::device>::value>;

template <typename E1, typename E2>
using EnableIfHostHost =
  std::enable_if_t<std::is_same<expr_space_type<E1>, space::host>::value &&
                   std::is_same<expr_space_type<E1>, space::host>::value>;

template <typename E1, typename E2>
using EnableIfHostDevice =
  std::enable_if_t<std::is_same<expr_space_type<E1>, space::host>::value &&
                   std::is_same<expr_space_type<E1>, space::host>::value>;

template <typename E1, typename E2>
using EnableIfDeviceHost =
  std::enable_if_t<std::is_same<expr_space_type<E1>, space::host>::value &&
                   std::is_same<expr_space_type<E1>, space::host>::value>;

} // namespace detail

// ----------------------------------------------------------------------
// ostream output

namespace detail
{

template <size_type N, typename S>
struct expression_printer;

template <size_type N>
struct expression_printer<N, space::host>
{
  template <typename E>
  static void print_to(std::ostream& os, const E& e_)
  {
    os << "{ shape: " << e_.shape() << "}\n";
  }
};

template <>
struct expression_printer<1, space::host>
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
struct expression_printer<2, space::host>
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

template <>
struct expression_printer<3, space::host>
{
  template <typename E>
  static void print_to(std::ostream& os, const E& e)
  {
    os << "{";
    for (int k = 0; k < e.shape(2); k++) {
      os << "{";
      for (int j = 0; j < e.shape(1); j++) {
        os << "{";
        for (int i = 0; i < e.shape(0); i++) {
          os << " " << e(i, j, k);
        }
        os << " }";
        if (j < e.shape(1) - 1) {
          os << "\n";
        }
      }
      os << " }";
      if (k < e.shape(2) - 1) {
        os << "\n";
      }
    }
    os << "}";
  }
};

#ifdef GTENSOR_HAVE_DEVICE

template <size_type N>
struct expression_printer<N, space::device>
{
  template <typename E>
  static void print_to(std::ostream& os, const E& e)
  {
    using gt_d =
      gtensor<expr_value_type<E>, expr_dimension<E>(), space::device>;
    using gt_h = gtensor<expr_value_type<E>, expr_dimension<E>(), space::host>;
    gt_d dtmp(e.shape());
    gt_h htmp(e.shape());
    dtmp = e;
    copy(dtmp, htmp);
    expression_printer<N, space::host>::print_to(os, htmp);
  }
};

#endif

} // namespace detail

template <typename E,
          typename Enable = std::enable_if_t<is_expression<E>::value>>
inline std::ostream& operator<<(std::ostream& os, const E& e)
{
  detail::expression_printer<expr_dimension<E>(), expr_space_type<E>>::print_to(
    os, e);
  return os;
}

// ----------------------------------------------------------------------
// operator==, !-
//
// immediate evaluation
// FIXME, should be done on device, too...

namespace detail
{

template <size_type N1, size_type N2, typename S1, typename S2>
struct equals
{
  template <typename E1, typename E2>
  static bool run(const E1& e1, const E2& e2)
  {
    return false;
  }
};

/*
template <size_type N>
struct equals<N, N>
{
  template <typename E1, typename E2, typename = EnableIfHostHost<E1, E2>>
  static bool run(const E1& e1_, const E2& e2_)
  {
    if (e1_.shape() != e2_.shape()) {
      std::cout << e1_.shape() << " != "
                << e2_.shape() << std::endl;
      return false;
    }
    auto e1 = linear_adapter(e1_);
    auto e2 = linear_adapter(e2_);
    std::cout << "e1 " << typeid(e1_).name() << " -> "
              << typeid(e1).name() << std::endl;
    std::cout << "e2 " << typeid(e2_).name() << " -> "
              << typeid(e2).name() << std::endl;
    std::cout << "e1 " << e1_.strides() " -> "
              <<  e1.strides() << std::endl;
    std::cout << "e2 " << e2_.strides() " -> "
              <<  e2.strides() << std::endl;
    for (int i = 0; i < calc_size(e1.shape()); i++) {
      if (e1.data_access(i) != e2.data_access(i)) {
        std::cout << e1.data_access(i) << " != "
                  << e2.data_access(i) << std::endl;
        return false;
      }
    }
    return true;
  }
}
*/

template <>
struct equals<1, 1, space::host, space::host>
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
struct equals<2, 2, space::host, space::host>
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
struct equals<3, 3, space::host, space::host>
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
struct equals<4, 4, space::host, space::host>
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
struct equals<5, 5, space::host, space::host>
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
struct equals<6, 6, space::host, space::host>
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

#ifdef GTENSOR_HAVE_DEVICE

template <size_type N1, size_type N2>
struct equals<N1, N2, space::device, space::device>
{
  template <typename E1, typename E2>
  static bool run(const E1& e1_, const E2& e2_)
  {
    if (e1_.shape() != e2_.shape()) {
      return false;
    }
    gtensor<expr_value_type<E1>, expr_dimension<E1>(), space::device> d1(
      e1_.shape());
    gtensor<expr_value_type<E2>, expr_dimension<E2>(), space::device> d2(
      e2_.shape());
    gtensor<expr_value_type<E1>, expr_dimension<E1>(), space::host> h1(
      e1_.shape());
    gtensor<expr_value_type<E2>, expr_dimension<E2>(), space::host> h2(
      e2_.shape());
    d1 = e1_;
    d2 = e2_;
    copy(d1, h1);
    copy(d2, h2);
    return equals<N1, N2, space::host, space::host>::run(h1, h2);
  }
};

template <size_type N1, size_type N2>
struct equals<N1, N2, space::device, space::host>
{
  template <typename E1, typename E2>
  static bool run(const E1& e1_, const E2& e2_)
  {
    if (e1_.shape() != e2_.shape()) {
      return false;
    }
    gtensor<expr_value_type<E1>, expr_dimension<E1>(), space::device> d1(
      e1_.shape());
    gtensor<expr_value_type<E1>, expr_dimension<E1>(), space::host> h1(
      e1_.shape());
    d1 = e1_;
    copy(d1, h1);
    return equals<N1, N2, space::host, space::host>::run(h1, e2_);
  }
};

template <size_type N1, size_type N2>
struct equals<N1, N2, space::host, space::device>
{
  template <typename E1, typename E2>
  static bool run(const E1& e1_, const E2& e2_)
  {
    if (e1_.shape() != e2_.shape()) {
      return false;
    }
    gtensor<expr_value_type<E2>, expr_dimension<E2>(), space::device> d2(
      e2_.shape());
    gtensor<expr_value_type<E2>, expr_dimension<E2>(), space::host> h2(
      e2_.shape());
    d2 = e2_;
    copy(d2, h2);
    return equals<N1, N2, space::host, space::host>::run(e1_, h2);
  }
};

#endif // GTENSOR_HAVE_DEVICE

} // namespace detail

template <typename E1, typename E2>
bool operator==(const expression<E1>& e1, const expression<E2>& e2)
{
  return detail::equals<E1::dimension(), E2::dimension(), expr_space_type<E1>,
                        expr_space_type<E2>>::run(e1.derived(), e2.derived());
}

template <typename E1, typename E2>
bool operator!=(const expression<E1>& e1, const expression<E2>& e2)
{
  return !(e1 == e2);
}

} // namespace gt

#endif
