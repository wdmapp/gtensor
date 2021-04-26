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

#endif // GTENSOR_USE_THRUST

#endif // GTENSOR_HAVE_DEVICE

} // namespace standard
} // namespace backend
} // namespace gt

#endif // GENSOR_DEVICE_COPY_H
