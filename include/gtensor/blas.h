#ifndef GTENSOR_BLAS_H
#define GTENSOR_BLAS_H

#include "gtensor/complex.h"
#include "gtensor/device_backend.h"
#include "gtensor/helper.h"
#include "gtensor/macros.h"
#include "gtensor/space.h"

namespace gt
{

namespace blas
{

// wrapper around backend specific type
struct handle_t;

} // namespace blas

} // namespace gt

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

template <typename C, typename = std::enable_if_t<has_container_methods_v<C> &&
                                                  has_space_type_device_v<C>>>
inline void axpy(handle_t* h, typename C::value_type a, const C& x, C& y)
{
  assert(x.size() == y.size());
  axpy(h, x.size(), a, gt::raw_pointer_cast(x.data()), 1,
       gt::raw_pointer_cast(y.data()), 1);
}

template <typename C, typename = std::enable_if_t<has_container_methods_v<C> &&
                                                  has_space_type_device_v<C>>>
inline void scal(handle_t* h, typename C::value_type fac, C& arr)
{
  scal(h, arr.size(), fac, gt::raw_pointer_cast(arr.data()), 1);
}

template <typename C, typename = std::enable_if_t<has_container_methods_v<C> &&
                                                  has_space_type_device_v<C> &&
                                                  has_complex_value_type_v<C>>>
inline void scal(handle_t* h, container_complex_subtype_t<C> fac, C& arr)
{
  scal(h, arr.size(), fac, gt::raw_pointer_cast(arr.data()), 1);
}

template <typename C, typename = std::enable_if_t<has_container_methods_v<C> &&
                                                  has_space_type_device_v<C>>>
inline void copy(handle_t* h, C& src, C& dest)
{
  copy(h, src.size(), gt::raw_pointer_cast(src.data()), 1,
       gt::raw_pointer_cast(dest.data()), 1);
}

template <typename C, typename = std::enable_if_t<has_container_methods_v<C> &&
                                                  has_space_type_device_v<C>>>
inline typename C::value_type dot(handle_t* h, const C& x, const C& y)
{
  return dot(h, x.size(), gt::raw_pointer_cast(x.data()), 1,
             gt::raw_pointer_cast(y.data()), 1);
}

template <typename C, typename = std::enable_if_t<has_container_methods_v<C> &&
                                                  has_space_type_device_v<C>>>
inline typename C::value_type dotu(handle_t* h, const C& x, const C& y)
{
  return dotu(h, x.size(), gt::raw_pointer_cast(x.data()), 1,
              gt::raw_pointer_cast(y.data()), 1);
}

template <typename C, typename = std::enable_if_t<has_container_methods_v<C> &&
                                                  has_space_type_device_v<C>>>
inline typename C::value_type dotc(handle_t* h, const C& x, const C& y)
{
  return dotc(h, x.size(), gt::raw_pointer_cast(x.data()), 1,
              gt::raw_pointer_cast(y.data()), 1);
}

template <typename M, typename V,
          typename = std::enable_if_t<
            has_container_methods_v<M> && has_space_type_device_v<M> &&
            has_container_methods_v<V> && has_space_type_device_v<V>>>
inline void gemv(handle_t* h, typename M::value_type alpha, M& A, V& x,
                 typename M::value_type beta, V& y)
{
  static_assert(expr_dimension<M>() == 2,
                "matrix arg 'A' must have dimension 2");
  static_assert(expr_dimension<V>() == 1,
                "vector args 'x' and 'y' must have dimension 1");
  static_assert(
    std::is_same<typename M::value_type, typename V::value_type>::value,
    "matrix and vectors must have same value type");
  assert(A.shape(1) == x.shape(0));

  gemv(h, A.shape(0), A.shape(1), alpha, gt::raw_pointer_cast(A.data()),
       A.shape(0), gt::raw_pointer_cast(x.data()), 1, beta,
       gt::raw_pointer_cast(y.data()), 1);
}

} // end namespace blas

} // end namespace gt

#endif
