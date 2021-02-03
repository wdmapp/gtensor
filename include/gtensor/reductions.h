#ifndef GTENSOR_REDUCTIONS_H
#define GTENSOR_REDUCTIONS_H

#include "defs.h"
#include "device_backend.h"

#include "gtensor_span.h"

#include "helper.h"

#ifdef GTENSOR_USE_THRUST

namespace gt
{

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

} // namespace gt

#endif // GTENSOR_USE_THRUST

#endif // GTENSOR_REDUCTIONS_H
