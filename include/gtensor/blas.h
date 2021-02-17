#ifndef GTENSOR_BLAS_H
#define GTENSOR_BLAS_H

#include "gtensor/complex.h"
#include "gtensor/device_backend.h"
#include "gtensor/helper.h"
#include "gtensor/macros.h"
#include "gtensor/space.h"

#if defined(GTENSOR_DEVICE_CUDA)
#include "gtensor/blas/cuda.h"
#elif defined(GTENSOR_DEVICE_HIP)
#include "gtensor/blas/hip.h"
#elif defined(GTENSOR_DEVICE_SYCL)
#include "gtensor/blas/sycl.h"
#endif

namespace gt
{

namespace blas
{

template <typename T, typename C,
          typename = std::enable_if_t<
            has_data_method_v<C> && has_size_method_v<C> &&
            std::is_same<expr_space_type<C>, gt::space::device>::value>>
inline void axpy(handle_t h, const T a, const C& x, C& y)
{
  static_assert(std::is_same<T, typename C::value_type>::value,
                "scalar 'a' must have same type as container value_type");

  assert(x.size() == y.size());

  axpy(h, x.size(), &a, gt::backend::raw_pointer_cast(x.data()), 1,
       gt::backend::raw_pointer_cast(y.data()), 1);
}

} // end namespace blas

} // end namespace gt

#endif
