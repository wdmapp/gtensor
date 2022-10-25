
#pragma once

#include <ISO_Fortran_binding.h>
#include <gtensor/gtensor.h>

namespace gt
{

template <typename T, std::size_t N>
struct farray
{
  CFI_CDESC_T(CFI_MAX_RANK) desc;
};

namespace detail
{

template <typename T>
struct CFI_type;

#define MAKE_CFI_type(T, CFI_T)                                                \
  template <>                                                                  \
  struct CFI_type<T>                                                           \
  {                                                                            \
    constexpr static int value = CFI_T;                                        \
  }

MAKE_CFI_type(float, CFI_type_float);
MAKE_CFI_type(double, CFI_type_double);
MAKE_CFI_type(gt::complex<float>, CFI_type_float_Complex);
MAKE_CFI_type(gt::complex<double>, CFI_type_double_Complex);

template <typename T, std::size_t N>
std::pair<gt::shape_type<N>, gt::shape_type<N>> to_shape_strides(
  farray<T, N>* nd)
{
  assert(nd->desc.rank == N);
  // complex type is broken in gfortran < 12 (?)
  // assert(nd->desc.type == detail::CFI_type<T>::value);
  gt::shape_type<N> shape, strides;
  for (int d = 0; d < N; d++) {
    shape[d] = nd->desc.dim[d].extent;
    strides[d] = (shape[d] == 1 ? 0 : nd->desc.dim[d].sm / sizeof(T));
  }
  return {shape, strides};
}

} // namespace detail

template <std::size_t N, typename T, typename S = space::host>
gtensor_span<T, N, S> adapt(farray<T, N>* nd)
{
  auto shape_strides = detail::to_shape_strides(nd);
  return gtensor_span<T, N, S>(static_cast<T*>(nd->desc.base_addr),
                               shape_strides.first, shape_strides.second);
}

template <std::size_t N, typename T>
gtensor_span<T, N, space::device> adapt_device(farray<T, N>* nd)
{
  auto shape_strides = detail::to_shape_strides(nd);
  return gtensor_span<T, N, space::device>(
    gt::device_pointer_cast(static_cast<T*>(nd->desc.base_addr)),
    shape_strides.first, shape_strides.second);
}

} // namespace gt
