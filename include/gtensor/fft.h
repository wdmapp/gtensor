#ifndef GTENSOR_FFT_H
#define GTENSOR_FFT_H

#include "complex.h"
#include "device_backend.h"
#include "helper.h"
#include "macros.h"
#include "space.h"

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
#include "fft/cuda.h"
#elif defined(GTENSOR_DEVICE_HIP)
#include "fft/hip.h"
#elif defined(GTENSOR_DEVICE_SYCL)
#include "fft/sycl.h"
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
  void operator()(const C1& in, C2& out) const
  {
    operator()(gt::backend::raw_pointer_cast(in.data()),
               gt::backend::raw_pointer_cast(out.data()));
  }

  template <typename C1, typename C2,
            typename = std::enable_if_t<
              has_container_methods_v<C1> && has_space_type_device_v<C1> &&
              has_container_methods_v<C2> && has_space_type_device_v<C2>>>
  void inverse(const C1& in, C2& out) const
  {
    inverse(gt::backend::raw_pointer_cast(in.data()),
            gt::backend::raw_pointer_cast(out.data()));
  }
};

} // namespace fft

} // namespace gt
#endif // GTENSOR_FFT_H
