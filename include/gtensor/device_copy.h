#ifndef GTENSOR_DEVICE_COPY_H
#define GTENSOR_DEVICE_COPY_H

#include <cstring>

#ifdef GTENSOR_HAVE_DEVICE
#include "device_backend.h"
#include "space.h"

#ifdef GTENSOR_USE_THRUST
#include "thrust_ext.h"
#endif

namespace gt {


#ifdef GTENSOR_USE_THRUST

template <typename T, typename S_from, typename S_to>
inline void memcpy(T *dest, const T *src, std::size_t bytes)
{
  thrust::copy(src, src + (bytes * sizeof(T)), dest);
}

#else // using gt::backend::device_storage

namespace detail {

template <typename S_from, typename S_to>
struct memcpy;

template <>
struct memcpy<space::device, space::host>
{
  static void run(void *dest, const void *src, std::size_t bytes)
  {
    gt::backend::device_memcpy_dh(dest, src, bytes);
  }
};

template <>
struct memcpy<space::device, space::device>
{
  static void run(void *dest, const void *src, std::size_t bytes)
  {
    gt::backend::device_memcpy_dd(dest, src, bytes);
  }
};

template <>
struct memcpy<space::host, space::device>
{
  static void run(void *dest, const void *src, std::size_t bytes)
  {
    gt::backend::device_memcpy_hd(dest, src, bytes);
  }
};

template <>
struct memcpy<space::host, space::host>
{
  static void run(void *dest, const void *src, std::size_t bytes)
  {
    std::memcpy(dest, src, bytes);
  }
};

} // end namespace detail

template <typename T, typename S_from, typename S_to>
inline void memcpy(T *dest, const T *src, std::size_t count)
{
  detail::memcpy<S_from, S_to>::run((void *)dest, (void *)src,
                                    count * sizeof(T));
}


#endif


} // end namespace gt

# else // not GTENSOR_HAVE_DEVICE

namespace gt {

template <typename T, typename S_from, typename S_to>
inline void memcpy(T *dest, const T *src, std::size_t count)
{
  std::memcpy((void *)dest, (void *)src, count * sizeof(T));
}

}

#endif // GTENSOR_HAVE_DEVICE

#endif // GENSOR_DEVICE_COPY_H
