
#ifndef GTENSOR_BACKEND_HOST_H
#define GTENSOR_BACKEND_HOST_H

#include "backend_common.h"

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

inline int device_get_count()
{
  return 1;
}

inline void device_set(int device_id)
{
  assert(device_id == 0);
}

inline int device_get()
{
  return 0;
}

inline uint32_t device_get_vendor_id(int device_id)
{
  return 0;
}

} // namespace host

namespace allocator_impl
{
template <typename T>
struct selector<T, gt::space::host_only>
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

#ifndef GTENSOR_HAVE_DEVICE

// streams are no-op on host (could use threads in the future)
class stream_view
{
public:
  stream_view() {}

  auto get_backend_stream() { return nullptr; }

  bool is_default() { return true; }

  void synchronize() {}
};

class stream
{
public:
  stream() {}

  auto get_backend_stream() { return nullptr; }

  bool is_default() { return true; }

  auto get_view() { return stream_view(); }

  void synchronize() {}
};

#endif

} // namespace gt

#endif // GTENSOR_BACKEND_HOST_H
