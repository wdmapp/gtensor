
#ifndef GTENSOR_FORWARD_H
#define GTENSOR_FORWARD_H

#include "defs.h"
#include "space.h"

namespace gt
{

template <typename EC, size_type N, typename S>
class gtensor_container;

template <typename T, size_type N, typename S = space::host>
using gtensor =
  gtensor_container<typename space::space_traits<S>::template storage_type<T>,
                    N, S>;
} // namespace gt

#endif
