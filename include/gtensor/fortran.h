
#pragma once

#include <gtensor/fortran/flcl.h>
#include <gtensor/gtensor.h>

namespace gt
{

namespace detail
{

template <std::size_t N>
auto to_shape(const flcl_ndarray_index_c_t* shape_in)
{
  shape_type<N> shape;
  for (std::size_t i = 0; i < N; i++) {
    shape[i] = shape_in[i];
  }
  return shape;
}

} // namespace detail

template <std::size_t N, typename T, typename S = space::host>
gtensor_span<T, N> adapt(const flcl_ndarray<T>* nd)
{
  assert(nd->rank == N);
  return gtensor_span<T, N>(static_cast<T*>(nd->data),
                            detail::to_shape<N>(nd->dims),
                            detail::to_shape<N>(nd->strides));
}

template <std::size_t N, typename T>
gtensor_span<T, N> adapt_device(const flcl_ndarray<T>* nd)
{
  return adapt<N, T, space::device>(nd);
}

} // namespace gt
