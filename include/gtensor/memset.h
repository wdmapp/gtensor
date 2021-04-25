#ifndef GTENSOR_MEMSET_H
#define GTENSOR_MEMSET_H

#include <cstring>

#include "device_backend.h"
#include "space.h"

namespace gt
{

namespace backend
{

template <typename S>
inline void memset(void* dst, int value, gt::size_type nbytes);

template <>
inline void memset<gt::space::host>(void* dst, int value, gt::size_type nbytes)
{
  std::memset(dst, value, nbytes);
}

#ifdef GTENSOR_HAVE_DEVICE

template <>
inline void memset<gt::space::device>(void* dst, int value,
                                      gt::size_type nbytes)
{
  gt::backend::standard::ops::memset(dst, value, nbytes);
}

#endif

} // namespace backend

} // namespace gt

#endif
