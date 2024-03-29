
#ifndef GTENSOR_FORWARD_H
#define GTENSOR_FORWARD_H

#include "defs.h"
#include "space.h"

namespace gt
{

namespace space
{
template <typename S>
struct space_traits2
{
#if GTENSOR_HAVE_DEVICE
  template <typename T>
  using storage_type = device_vector<T>;
#endif
};

template <>
struct space_traits2<host>
{
  template <typename T>
  using storage_type = host_vector<T>;
};

} // namespace space

template <typename EC, size_type N>
class gtensor_container;

template <typename T, size_type N, typename S = space::host>
using gtensor =
  gtensor_container<typename space::space_traits2<S>::template storage_type<T>,
                    N>;
} // namespace gt

#endif
