#ifndef GTENSOR_REDUCTIONS_H
#define GTENSOR_REDUCTIONS_H

#include "gtensor.h"

#include <assert.h>
#include <type_traits>

//#include <iostream>

namespace gt
{

#ifdef GTENSOR_USE_THRUST

template <typename Container,
          typename = std::enable_if_t<has_data_method_v<Container>>>
inline auto sum(const Container& a)
{
  using T = typename Container::value_type;
  auto begin = a.data();
  auto end = a.data() + a.size();
  return thrust::reduce(begin, end, 0., thrust::plus<T>());
}

template <typename Container,
          typename = std::enable_if_t<has_data_method_v<Container>>>
inline auto max(const Container& a)
{
  using T = typename Container::value_type;
  auto begin = a.data();
  auto end = a.data() + a.size();
  return thrust::reduce(begin, end, 0., thrust::maximum<T>());
}

template <typename Container,
          typename = std::enable_if_t<has_data_method_v<Container>>>
inline auto min(const Container& a)
{
  auto begin = a.data();
  auto end = a.data() + a.size();
  auto min_element = thrust::min_element(begin, end);
  return *min_element;
}

#endif // GTENSOR_USE_THRUST

template <typename Eout, typename Ein>
inline void sum_axis_to(Eout&& out, Ein&& in, int axis)
{
  using Sout = expr_space_type<Eout>;
  using Tout = expr_value_type<Eout>;
  using Sin = expr_space_type<Ein>;
  using Tin = expr_value_type<Ein>;
  using shape_type = expr_shape_type<Eout>;

  static_assert(std::is_same<Sout, Sin>::value,
                "out and in expressions must be in the same space");
  static_assert(std::is_same<Tout, Tin>::value,
                "out and in expressions must have the same value type");

  constexpr auto dims_out = expr_dimension<Eout>();
  constexpr auto dims_in = expr_dimension<Ein>();

  static_assert(
    dims_out == dims_in - 1,
    "out expression must have one less dimension than in expression");

  auto shape_in = in.shape();
  auto shape_out = out.shape();
  auto shape_out_expected = shape_in.remove(axis);

  assert(shape_out == shape_out_expected);

  auto k_out = out.to_kernel();
  auto k_in = in.to_kernel();

  // Note: use logical indexing strides, not internal strides which may be
  // for addressing the underlying data for gview
  auto strides_out = calc_strides(shape_out);
  auto strides_in = calc_strides(shape_in);

  auto flat_out_shape = gt::shape(static_cast<int>(out.size()));
  int reduction_length = in.shape(axis);

  gt::launch<1, Sout>(
    flat_out_shape, GT_LAMBDA(int i) {
      auto idx_out = unravel(i, strides_out);
      auto idx_in = idx_out.insert(axis, 0);
      Tin tmp = index_expression(k_in, idx_in);
      idx_in[axis]++;
      for (int j = 1; j < reduction_length; j++) {
        tmp = tmp + index_expression(k_in, idx_in);
        idx_in[axis]++;
      }
      index_expression(k_out, idx_out) = tmp;
    });
}

} // namespace gt

#endif // GTENSOR_REDUCTIONS_H
