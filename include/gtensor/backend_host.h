
#ifndef GTENSOR_BACKEND_HOST_H
#define GTENSOR_BACKEND_HOST_H

#include "pointer_traits.h"

#include <algorithm>

// ======================================================================
// gt::backend::host

namespace gt
{
namespace backend
{

namespace host
{

inline void device_synchronize()
{
  // no need to synchronize on host
}

} // namespace host

namespace allocator_impl
{
template <typename T>
struct selector<T, gt::space::host>
{
  using type = std::allocator<T>;
};
} // namespace allocator_impl

namespace copy_impl
{
template <typename InputPtr, typename OutputPtr>
inline void copy_n(gt::space::host tag_in, gt::space::host tag_out, InputPtr in,
                   size_type count, OutputPtr out)
{
  std::copy_n(in, count, out);
}
} // namespace copy_impl

namespace fill_impl
{
template <typename Ptr, typename T>
inline void fill(gt::space::host tag, Ptr first, Ptr last, const T& value)
{
  std::fill(first, last, value);
}
} // namespace fill_impl

} // namespace backend
} // namespace gt

#endif // GTENSOR_BACKEND_HOST_H
