#ifndef GTENSOR_FFT_H
#define GTENSOR_FFT_H

#include "gtensor/complex.h"
#include "gtensor/device_backend.h"
#include "gtensor/helper.h"
#include "gtensor/macros.h"
#include "gtensor/space.h"

namespace gt
{

namespace fft
{

enum class Domain
{
  REAL,
  COMPLEX
};

} // namespace fft

} // namespace gt

#if defined(GTENSOR_DEVICE_CUDA)
#include "backend/cuda.h"
#elif defined(GTENSOR_DEVICE_HIP)
#include "backend/hip.h"
#elif defined(GTENSOR_DEVICE_SYCL)
#include "backend/sycl.h"
#endif

namespace gt
{

namespace fft
{

template <gt::fft::Domain D, typename R>
class FFTPlanMany : public FFTPlanManyBackend<D, R>
{
public:
  using FFTPlanManyBackend<D, R>::FFTPlanManyBackend;

  using FFTPlanManyBackend<D, R>::operator();
  using FFTPlanManyBackend<D, R>::inverse;

  template <typename C1, typename C2,
            typename = std::enable_if_t<
              has_container_methods_v<C1> && has_space_type_device_v<C1> &&
              has_container_methods_v<C2> && has_space_type_device_v<C2>>>
  void operator()(C1& in, C2& out) const
  {
    operator()(gt::raw_pointer_cast(in.data()),
               gt::raw_pointer_cast(out.data()));
  }

  template <typename C1, typename C2,
            typename = std::enable_if_t<
              has_container_methods_v<C1> && has_space_type_device_v<C1> &&
              has_container_methods_v<C2> && has_space_type_device_v<C2>>>
  void inverse(C1& in, C2& out) const
  {
    inverse(gt::raw_pointer_cast(in.data()), gt::raw_pointer_cast(out.data()));
  }
};

} // namespace fft

} // namespace gt
#endif // GTENSOR_FFT_H
