#ifndef GTENSOR_REDUCTIONS_H
#define GTENSOR_REDUCTIONS_H

#include "gtensor.h"

#if defined(GTENSOR_DEVICE_CUDA) || defined(GTENSOR_DEVICE_HIP)
#include <thrust/reduce.h>
#endif

#include <cassert>
#include <functional>
#include <numeric>
#include <type_traits>

//#include <iostream>

namespace gt
{

#if defined(GTENSOR_DEVICE_CUDA) || defined(GTENSOR_DEVICE_HIP)

namespace detail
{

template <typename Container, typename Enable = void>
struct thrust_const_pointer
{
  using type = typename Container::const_pointer;
};

template <typename Container>
struct thrust_const_pointer<
  Container, gt::meta::void_t<std::enable_if_t<std::is_same<
               typename Container::space_type, gt::space::device>::value>>>
{
  using type =
    thrust::device_ptr<std::add_const_t<typename Container::value_type>>;
};

} // namespace detail

template <typename Container,
          typename = std::enable_if_t<has_data_method_v<Container>>>
inline auto sum(const Container& a)
{
  using T = typename Container::value_type;
  using P = typename detail::thrust_const_pointer<Container>::type;
  // Note: wrapping in device_ptr before passing to reduce is necessary
  // when using the non-thrust storage backend, for HIP and CUDA 10.2.
  // Not necessary in CUDA 11.2. For thrust backend and newer CUDA,
  // this only entails an extra device_ptr copy construct, so performance
  // impact will be minimal.
  P begin(gt::raw_pointer_cast(a.data()));
  P end(gt::raw_pointer_cast(a.data()) + a.size());
  return thrust::reduce(begin, end, 0., thrust::plus<T>());
}

template <typename Container,
          typename = std::enable_if_t<has_data_method_v<Container>>>
inline auto max(const Container& a)
{
  using T = typename Container::value_type;
  using P = typename detail::thrust_const_pointer<Container>::type;
  P begin(gt::raw_pointer_cast(a.data()));
  P end(gt::raw_pointer_cast(a.data()) + a.size());
  return thrust::reduce(begin, end, 0., thrust::maximum<T>());
}

template <typename Container,
          typename = std::enable_if_t<has_data_method_v<Container>>>
inline auto min(const Container& a)
{
  using T = typename Container::value_type;
  using P = typename detail::thrust_const_pointer<Container>::type;
  P begin(gt::raw_pointer_cast(a.data()));
  P end(gt::raw_pointer_cast(a.data()) + a.size());
  auto min_element = thrust::min_element(begin, end);
  return *min_element;
}

template <typename Container, typename OutputType, typename BinaryReductionOp,
          typename = std::enable_if_t<has_data_method_v<Container>>>
inline OutputType reduce(const Container& a, OutputType init,
                         BinaryReductionOp reduction_op)
{
  using T = typename Container::value_type;
  using P = typename detail::thrust_const_pointer<Container>::type;
  // Note: wrapping in device_ptr before passing to reduce is necessary
  // when using the non-thrust storage backend, for HIP and CUDA 10.2.
  // Not necessary in CUDA 11.2. For thrust backend and newer CUDA,
  // this only entails an extra device_ptr copy construct, so performance
  // impact will be minimal.
  P begin(gt::raw_pointer_cast(a.data()));
  P end(gt::raw_pointer_cast(a.data()) + a.size());
  return thrust::reduce(begin, end, init, reduction_op);
}

template <typename Container, typename OutputType, typename BinaryReductionOp,
          typename UnaryTransformOp,
          typename = std::enable_if_t<has_data_method_v<Container>>>
inline OutputType transform_reduce(const Container& a, OutputType init,
                                   BinaryReductionOp reduction_op,
                                   UnaryTransformOp transform_op)
{
  using T = typename Container::value_type;
  using P = typename detail::thrust_const_pointer<Container>::type;
  // Note: wrapping in device_ptr before passing to reduce is necessary
  // when using the non-thrust storage backend, for HIP and CUDA 10.2.
  // Not necessary in CUDA 11.2. For thrust backend and newer CUDA,
  // this only entails an extra device_ptr copy construct, so performance
  // impact will be minimal.
  P begin(gt::raw_pointer_cast(a.data()));
  P end(gt::raw_pointer_cast(a.data()) + a.size());
  return thrust::transform_reduce(begin, end, transform_op, init, reduction_op);
}

#elif defined(GTENSOR_DEVICE_SYCL)

#ifdef GTENSOR_DEVICE_SYCL_HOST
#define SYCL_REDUCTION_WORK_GROUP_SIZE 16
#else
#define SYCL_REDUCTION_WORK_GROUP_SIZE 128
#endif

template <typename Container,
          typename = std::enable_if_t<
            has_data_method_v<Container> &&
            std::is_same<typename Container::space_type, space::device>::value>>
inline auto sum(const Container& a)
{
  using T = typename Container::value_type;
  sycl::queue& q = gt::backend::sycl::get_queue();
  T sum_result = 0;
  sycl::buffer<T> sum_buf{&sum_result, 1};
  {
    sycl::nd_range<1> range(a.size(), SYCL_REDUCTION_WORK_GROUP_SIZE);
    if (a.size() <= SYCL_REDUCTION_WORK_GROUP_SIZE) {
      range = sycl::nd_range<1>(a.size(), a.size());
    }
    auto data = a.data();
    auto e = q.submit([&](sycl::handler& cgh) {
      auto sum_acc =
        sum_buf.template get_access<sycl::access::mode::read_write>(cgh);
      auto sum_reducer =
        sycl::ext::oneapi::reduction(sum_acc, sycl::ext::oneapi::plus<T>{});
      using kname = gt::backend::sycl::Sum<Container>;
      cgh.parallel_for<kname>(range, sum_reducer,
                              [=](sycl::nd_item<1> item, auto& sum) {
                                sum += data[item.get_global_id(0)];
                              });
    });
    e.wait();
  }
  return sum_buf.get_host_access()[0];
}

template <typename Container,
          typename = std::enable_if_t<
            has_data_method_v<Container> &&
            std::is_same<typename Container::space_type, space::device>::value>>
inline auto max(const Container& a)
{
  using T = typename Container::value_type;
  sycl::queue& q = gt::backend::sycl::get_queue();
  std::array<T, 1> result;
  T max_result = 0;
  sycl::buffer<T> max_buf{&max_result, 1};
  {
    sycl::nd_range<1> range(a.size(), SYCL_REDUCTION_WORK_GROUP_SIZE);
    if (a.size() <= SYCL_REDUCTION_WORK_GROUP_SIZE) {
      range = sycl::nd_range<1>(a.size(), a.size());
    }
    auto data = a.data();
    auto e = q.submit([&](sycl::handler& cgh) {
      auto max_acc =
        max_buf.template get_access<sycl::access::mode::read_write>(cgh);
      auto max_reducer =
        sycl::ext::oneapi::reduction(max_acc, sycl::ext::oneapi::maximum<T>{});
      using kname = gt::backend::sycl::Max<Container>;
      cgh.parallel_for<kname>(range, max_reducer,
                              [=](sycl::nd_item<1> item, auto& max) {
                                max.combine(data[item.get_global_id(0)]);
                              });
    });
    e.wait();
  }
  return max_buf.get_host_access()[0];
}

template <typename Container,
          typename = std::enable_if_t<
            has_data_method_v<Container> &&
            std::is_same<typename Container::space_type, space::device>::value>>
inline auto min(const Container& a)
{
  using T = typename Container::value_type;
  sycl::queue& q = gt::backend::sycl::get_queue();
  T min_result;
  sycl::buffer<T> min_buf{&min_result, 1};
  {
    sycl::nd_range<1> range(a.size(), SYCL_REDUCTION_WORK_GROUP_SIZE);
    if (a.size() <= SYCL_REDUCTION_WORK_GROUP_SIZE) {
      range = sycl::nd_range<1>(a.size(), a.size());
    }
    auto data = a.data();
    // TODO: this is hacky, there must be a better way?
    q.submit([&](sycl::handler& cgh) {
      auto min_write =
        min_buf.template get_access<sycl::access::mode::discard_write>(cgh);
      cgh.single_task([=]() { min_write[0] = data[0]; });
    });
    auto e = q.submit([&](sycl::handler& cgh) {
      auto min_acc =
        min_buf.template get_access<sycl::access::mode::read_write>(cgh);
      auto min_reducer =
        sycl::ext::oneapi::reduction(min_acc, sycl::ext::oneapi::minimum<T>{});
      using kname = gt::backend::sycl::Min<Container>;
      cgh.parallel_for<kname>(range, min_reducer,
                              [=](sycl::nd_item<1> item, auto& min) {
                                min.combine(data[item.get_global_id(0)]);
                              });
    });
    e.wait();
  }
  return min_buf.get_host_access()[0];
}

namespace detail
{

template <typename T>
struct UnaryOpIdentity
{
  GT_INLINE T operator()(T a) const { return a; }
};

} // namespace detail

template <typename Container, typename OutputType, typename BinaryReductionOp,
          typename = std::enable_if_t<
            has_data_method_v<Container> &&
            std::is_same<typename Container::space_type, space::device>::value>>
inline OutputType reduce(const Container& a, OutputType init,
                         BinaryReductionOp reduction_op)
{
  using ValueType = typename Container::value_type;
  auto identity_op = detail::UnaryOpIdentity<ValueType>{};
  return transform_reduce(a, init, reduction_op, identity_op);
}

template <typename Container, typename OutputType, typename BinaryReductionOp,
          typename UnaryTransformOp,
          typename = std::enable_if_t<
            has_data_method_v<Container> &&
            std::is_same<typename Container::space_type, space::device>::value>>
inline OutputType transform_reduce(const Container& a, OutputType init,
                                   BinaryReductionOp reduction_op,
                                   UnaryTransformOp transform_op)
{
  sycl::queue& q = gt::backend::sycl::get_queue();
  OutputType result = init;
  sycl::buffer<OutputType> result_buf{&result, 1};
  {
    sycl::nd_range<1> range(a.size(), SYCL_REDUCTION_WORK_GROUP_SIZE);
    if (a.size() <= SYCL_REDUCTION_WORK_GROUP_SIZE) {
      range = sycl::nd_range<1>(a.size(), a.size());
    }
    auto data = a.data();
    auto e = q.submit([&](sycl::handler& cgh) {
      auto result_acc =
        result_buf.template get_access<sycl::access::mode::read_write>(cgh);
      auto reducer =
        sycl::ext::oneapi::reduction(result_acc, init, reduction_op);
      cgh.parallel_for(range, reducer, [=](sycl::nd_item<1> item, auto& r) {
        r.combine(transform_op(data[item.get_global_id(0)]));
      });
    });
    e.wait();
  }
  return result_buf.get_host_access()[0];
}

#endif // device implementations

#if defined(GTENSOR_DEVICE_HOST) || defined(GTENSOR_DEVICE_SYCL)

template <typename Container,
          typename = std::enable_if_t<
            has_data_method_v<Container> &&
            std::is_same<typename Container::space_type, space::host>::value>,
          typename = int>
inline auto sum(const Container& a)
{
  using T = typename Container::value_type;
  auto data = a.data();
  // TODO: this assumes type has an initializer from int(0), which should be
  // true for all numeric types encountered in practice, but this is ugly
  T tmp = 0;
  for (int i = 0; i < a.size(); i++) {
    tmp += data[i];
  }
  return tmp;
}

template <typename Container,
          typename = std::enable_if_t<
            has_data_method_v<Container> &&
            std::is_same<typename Container::space_type, space::host>::value>,
          typename = int>
inline auto max(const Container& a)
{
  using T = typename Container::value_type;
  auto data = a.data();
  T max_value = data[0];
  T current_value;
  for (int i = 1; i < a.size(); i++) {
    current_value = data[i];
    if (current_value > max_value)
      max_value = current_value;
  }
  return max_value;
}

template <typename Container,
          typename = std::enable_if_t<
            has_data_method_v<Container> &&
            std::is_same<typename Container::space_type, space::host>::value>,
          typename = int>
inline auto min(const Container& a)
{
  using T = typename Container::value_type;
  auto data = a.data();
  T min_value = data[0];
  T current_value;
  for (int i = 0; i < a.size(); i++) {
    current_value = data[i];
    if (current_value < min_value)
      min_value = current_value;
  }
  return min_value;
}

template <typename Container, typename OutputType, typename BinaryReductionOp,
          typename = std::enable_if_t<
            has_data_method_v<Container> &&
            std::is_same<typename Container::space_type, space::host>::value>,
          typename = int>
inline OutputType reduce(const Container& a, OutputType init,
                         BinaryReductionOp reduction_op)
{
  using P = const typename Container::value_type*;
  P begin(a.data());
  P end(a.data() + a.size());
  return std::accumulate(begin, end, init, reduction_op);
}

template <typename Container, typename OutputType, typename BinaryReductionOp,
          typename UnaryTransformOp,
          typename = std::enable_if_t<
            has_data_method_v<Container> &&
            std::is_same<typename Container::space_type, space::host>::value>,
          typename = int>
inline OutputType transform_reduce(const Container& a, OutputType init,
                                   BinaryReductionOp reduction_op,
                                   UnaryTransformOp transform_op)
{
  using P = const typename Container::value_type*;
  P begin(a.data());
  P end(a.data() + a.size());
#if __cplusplus >= 201703L
  return std::transform_reduce(begin, end, init, reduction_op, transform_op);
#else
  for (P dp = begin; dp < end; dp++) {
    init = reduction_op(init, transform_op(*dp));
  }
  return init;
#endif
}

#endif // HOST / SYCL host implementation

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
  auto shape_out_expected = remove(shape_in, axis);

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
      auto idx_in = insert(idx_out, axis, 0);
      Tin tmp = k_in[idx_in];
      idx_in[axis]++;
      for (int j = 1; j < reduction_length; j++) {
        tmp = tmp + k_in[idx_in];
        idx_in[axis]++;
      }
      k_out[idx_out] = tmp;
    });
}

template <typename E>
auto norm_linf(const E& e)
{
  // FIXME, the gt::eval is a workaround for gt::max only handling containers
  return gt::max(gt::eval(gt::abs(e)));
}

namespace detail
{

template <typename Tin, typename Tout = Tin, typename Enable = void>
struct UnaryOpNorm
{
  GT_INLINE Tout operator()(Tin a) const { return a * a; }
};

template <typename Tin, typename Tout>
struct UnaryOpNorm<Tin, Tout,
                   gt::meta::void_t<std::enable_if_t<gt::is_complex_v<Tin>>>>
{
  GT_INLINE Tout operator()(Tin a) const { return gt::norm(a); }
};

} // namespace detail

/*! Reduction helper implementing sum of squares on arbitrary expressions. For
 * complex valued arrays, uses `gt::norm` instead of square, so it calculates
 * the L2 norm squared.
 *
 * If e is not a container type, it will be evaluated into a temporary array, so
 * this may not be the most efficient approach in some cases. This is a
 * limitation of the current reduction implementations using low level pointers,
 * passed down to thrust backend instead of logical gtensor expression index
 * pointers.
 */
template <typename E>
auto sum_squares(const E& e)
{
  // FIXME, the gt::eval is a workaround for gt::transform_reduce only handling
  // containers
  using ValueType = expr_value_type<E>;
  using Real = gt::complex_subtype_t<ValueType>;
  return gt::transform_reduce(gt::eval(e), 0.0, std::plus<>{},
                              detail::UnaryOpNorm<ValueType, Real>{});
}

} // namespace gt

#endif // GTENSOR_REDUCTIONS_H
