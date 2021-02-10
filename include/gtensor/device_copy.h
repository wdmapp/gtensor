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

namespace detail
{

template <typename S_from, typename S_to>
struct copy;

template <>
struct copy<space::device, space::host>
{
  template <typename T>
  static void run(const T* src, T* dest, std::size_t count)
  {
    gt::backend::device_copy_dh(src, dest, count);
  }
};

template <>
struct copy<space::device, space::device>
{
  template <typename T>
  static void run(const T* src, T* dest, std::size_t count)
  {
    gt::backend::device_copy_dd(src, dest, count);
  }
};

template <>
struct copy<space::host, space::device>
{
  template <typename T>
  static void run(const T* src, T* dest, std::size_t count)
  {
    gt::backend::device_copy_hd(src, dest, count);
  }
};

template <>
struct copy<space::host, space::host>
{
  template <typename T>
  static void run(const T* src, T* dest, std::size_t count)
  {
    gt::backend::device_copy_hh(src, dest, count);
  }
};

} // end namespace detail

template <typename T, typename S_from, typename S_to>
inline void copy(const T* src, T* dest, std::size_t count)
{
  detail::copy<S_from, S_to>::run(src, dest, count);
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
