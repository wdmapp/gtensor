#ifndef GTENSOR_DEVICE_COPY_H
#define GTENSOR_DEVICE_COPY_H

#include <cstring>

#ifdef GTENSOR_HAVE_DEVICE
#include "device_backend.h"
#include "space.h"

#ifdef GTENSOR_USE_THRUST
#include "thrust_ext.h"
#endif

namespace gt
{

namespace backend
{

#ifdef GTENSOR_USE_THRUST

template <typename T, typename S_from, typename S_to>
inline void copy(const T* src, T* dest, std::size_t count)
{
  thrust::copy(src, src + count, dest);
}

template <typename T, typename S_from, typename S_to>
inline void copy(thrust::device_ptr<const T> src, T* dest, std::size_t count)
{
  thrust::copy(src, src + count, dest);
}

template <typename T, typename S_from, typename S_to>
inline void copy(const T* src, thrust::device_ptr<T> dest, std::size_t count)
{
  thrust::copy(src, src + count, dest);
}

template <typename T, typename S_from, typename S_to>
inline void copy(thrust::device_ptr<const T> src, thrust::device_ptr<T> dest,
                 std::size_t count)
{
  thrust::copy(src, src + count, dest);
}

#else // using gt::backend::device_storage

template <typename T, typename S_from, typename S_to>
inline void copy(const T* src, T* dest, std::size_t count)
{
  gt::backend::ops::copy<S_from, S_to>(src, dest, count);
}

#endif

} // end namespace backend

} // end namespace gt

#else // not GTENSOR_HAVE_DEVICE

namespace gt
{
namespace backend
{

template <typename T, typename S_from, typename S_to>
inline void copy(const T* src, T* dest, std::size_t count)
{
  std::memcpy((void*)dest, (void*)src, count * sizeof(T));
}

} // namespace backend
} // namespace gt

#endif // GTENSOR_HAVE_DEVICE

#endif // GENSOR_DEVICE_COPY_H
