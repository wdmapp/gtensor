#ifndef GTENSOR_DEVICE_COPY_H
#define GTENSOR_DEVICE_COPY_H

#include <cstring>

#ifdef GTENSOR_HAVE_DEVICE

#include "device_backend.h"

#ifdef GTENSOR_USE_THRUST
#include "thrust_ext.h"
#endif

#endif // GTENSOR_HAVE_DEVICE

namespace gt
{
namespace backend
{
namespace standard
{

#ifdef GTENSOR_HAVE_DEVICE
#ifdef GTENSOR_USE_THRUST

#else // !GTENSOR_USE_THRUST: using gt::backend::device_storage

template <typename S_from, typename S_to, typename T>
inline void copy(const T* src, T* dest, std::size_t count)
{
  gt::backend::standard::ops::copy<S_from, S_to>(src, dest, count);
}

#endif // GTENSOR_USE_THRUST

#endif // GTENSOR_HAVE_DEVICE

} // namespace standard
} // namespace backend
} // namespace gt

#endif // GENSOR_DEVICE_COPY_H
