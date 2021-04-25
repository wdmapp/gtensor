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

template <typename S_from, typename S_to, typename T>
inline void copy(const T* src, T* dest, std::size_t count)
{
  thrust::copy(src, src + count, dest);
}

template <typename S_from, typename S_to, typename T>
inline void copy(thrust::device_ptr<const T> src, T* dest, std::size_t count)
{
  thrust::copy(src, src + count, dest);
}

template <typename S_from, typename S_to, typename T>
inline void copy(thrust::device_ptr<T> src, T* dest, std::size_t count)
{
  thrust::copy(src, src + count, dest);
}

template <typename S_from, typename S_to, typename T>
inline void copy(const T* src, thrust::device_ptr<T> dest, std::size_t count)
{
  thrust::copy(src, src + count, dest);
}

template <typename S_from, typename S_to, typename T>
inline void copy(thrust::device_ptr<const T> src, thrust::device_ptr<T> dest,
                 std::size_t count)
{
  thrust::copy(src, src + count, dest);
}

#else // !GTENSOR_USE_THRUST: using gt::backend::device_storage

template <typename S_from, typename S_to, typename T>
inline void copy(const T* src, T* dest, std::size_t count)
{
  gt::backend::ops::copy<S_from, S_to>(src, dest, count);
}

#endif // GTENSOR_USE_THRUST

#else // not GTENSOR_HAVE_DEVICE

template <typename S_from, typename S_to, typename T>
inline void copy(const T* src, T* dest, std::size_t count)
{
  std::memcpy((void*)dest, (void*)src, count * sizeof(T));
}

#endif // GTENSOR_HAVE_DEVICE

} // namespace standard
} // namespace backend
} // namespace gt

#endif // GENSOR_DEVICE_COPY_H
