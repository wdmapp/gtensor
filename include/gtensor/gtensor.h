
#ifndef GTENSOR_GTENSOR_H
#define GTENSOR_GTENSOR_H

#ifdef __CUDACC__
#include "thrust_ext.h"
#endif

#include "gfunction.h"
#include "gtensor_view.h"
#include "gview.h"

namespace gt
{

// ======================================================================
// gtensor

template <typename T, int N, typename S>
class gtensor;

template <typename T, int N, typename S>
struct gtensor_inner_types<gtensor<T, N, S>>
{
  using space_type = S;
  constexpr static size_type dimension = N;

  using storage_type = typename space_type::template Vector<T>;
  using value_type = typename storage_type::value_type;
  using pointer = typename storage_type::pointer;
  using const_pointer = typename storage_type::const_pointer;
  using reference = typename storage_type::reference;
  using const_reference = typename storage_type::const_reference;
};

template <typename T, int N, typename S = space::host>
class gtensor : public gcontainer<gtensor<T, N, S>>
{
public:
  using self_type = gtensor<T, N, S>;
  using base_type = gcontainer<self_type>;
  using inner_types = gtensor_inner_types<self_type>;
  using storage_type = typename inner_types::storage_type;

  using typename base_type::const_pointer;
  using typename base_type::const_reference;
  using typename base_type::pointer;
  using typename base_type::reference;
  using typename base_type::shape_type;
  using typename base_type::strides_type;
  using typename base_type::value_type;

  using base_type::dimension;

  using base_type::base_type;
  gtensor() = default;
  explicit gtensor(const shape_type& shape);
  gtensor(helper::nd_initializer_list_t<T, N> il);
  template <typename E>
  gtensor(const expression<E>& e);

  using base_type::operator=;

  gtensor_view<T, N, S> to_kernel() const; // FIXME, const T
  gtensor_view<T, N, S> to_kernel();

private:
  GT_INLINE const storage_type& storage_impl() const;
  GT_INLINE storage_type& storage_impl();
  GT_INLINE const_reference data_access_impl(size_type i) const;
  GT_INLINE reference data_access_impl(size_type i);

  storage_type storage_;

  friend class gstrided<self_type>;
  friend class gcontainer<self_type>;
};

// ======================================================================
// gtensor implementation

template <typename T, int N, typename S>
inline gtensor<T, N, S>::gtensor(const shape_type& shape)
  : base_type(shape, calc_strides(shape)), storage_(calc_size(shape))
{}

template <typename T, int N, typename S>
inline gtensor<T, N, S>::gtensor(helper::nd_initializer_list_t<T, N> il)
  : base_type({}, {})
{
  // FIXME?! this kinda changes row-major list into transposed col-major array
  shape_type shape = helper::nd_initializer_list_shape<N>(il);
  base_type::resize(shape);
  helper::nd_initializer_list_copy<N>(il, (*this));
}

template <typename T, int N, typename S>
template <typename E>
inline gtensor<T, N, S>::gtensor(const expression<E>& e)
{
  this->resize(e.derived().shape());
  *this = e.derived();
}

template <typename T, int N, typename S>
inline auto gtensor<T, N, S>::storage_impl() const -> const storage_type&
{
  return storage_;
}

template <typename T, int N, typename S>
inline auto gtensor<T, N, S>::storage_impl() -> storage_type&
{
  return storage_;
}

#pragma nv_exec_check_disable
template <typename T, int N, typename S>
inline auto gtensor<T, N, S>::data_access_impl(size_t i) const
  -> const_reference
{
  return storage_[i];
}

#pragma nv_exec_check_disable
template <typename T, int N, typename S>
inline auto gtensor<T, N, S>::data_access_impl(size_t i) -> reference
{
  return storage_[i];
}

template <typename T, int N, typename S>
inline gtensor_view<T, N, S> gtensor<T, N, S>::to_kernel() const
{
  return gtensor_view<T, N, S>(const_cast<gtensor<T, N, S>*>(this)->data(),
                               this->shape(), this->strides());
}

template <typename T, int N, typename S>
inline gtensor_view<T, N, S> gtensor<T, N, S>::to_kernel()
{
  return gtensor_view<T, N, S>(this->data(), this->shape(), this->strides());
}

#ifdef __CUDACC__

// ======================================================================
// copies
//
// FIXME, there should be only one, more general version,
// and maybe this should be .assign or operator=

template <typename T, int N, typename S>
void copy(const gtensor_view<T, N>& from, gtensor<T, N, S>& to)
{
  assert(from.size() == to.size());
  thrust::copy(from.data(), from.data() + from.size(), to.data());
}

template <typename T, int N, typename S>
void copy(const gtensor<T, N, S>& from, gtensor_view<T, N>& to)
{
  assert(from.size() == to.size());
  thrust::copy(from.data(), from.data() + from.size(), to.data());
}

template <typename T, int N, typename S_from, typename S_to>
void copy(const gtensor<T, N, S_from>& from, gtensor<T, N, S_to>& to)
{
  assert(from.size() == to.size());
  thrust::copy(from.data(), from.data() + from.size(), to.data());
}

// ======================================================================
// launch

template <typename F>
__global__ void kernel_launch(gt::shape_type<3> shape, F f)
{
  int i = threadIdx.x + blockIdx.x * BS_X;
  int j = threadIdx.y + blockIdx.y * BS_Y;
  int b = blockIdx.z;

  if (i < shape[0] && j < shape[1]) {
    f(i, j, b);
  }
}

template <int N, typename F>
inline void launch(const gt::shape_type<3>& shape, F&& f)
{
  static_assert(N == 3, "launch only supports 3-dims");
  dim3 numThreads(BS_X, BS_Y);
  dim3 numBlocks((shape[0] + BS_X - 1) / BS_X, (shape[1] + BS_Y - 1) / BS_Y,
                 shape[2]);

  kernel_launch<<<numBlocks, numThreads>>>(shape, std::forward<F>(f));
}

#endif

// ======================================================================
// gtensor_device, gtensor_view_device

template <typename T, int N>
using gtensor_device = gtensor<T, N, space::device>;

template <typename T, int N>
using gtensor_view_device = gtensor_view<T, N, space::device>;

// ======================================================================
// empty_like

template <typename E>
inline auto empty_like(const expression<E>& _e)
{
  const auto& e = _e.derived();
  return gtensor<expr_value_type<E>, expr_dimension<E>(), expr_space_type<E>>(
    e.shape());
}

// ======================================================================
// zeros_like

template <typename E>
inline auto zeros_like(const expression<E>& _e)
{
  const auto& e = _e.derived();
  return gtensor<expr_value_type<E>, expr_dimension<E>()>(e.shape());
}

// ======================================================================
// eval

template <typename E>
using is_gcontainer = std::is_base_of<gcontainer<E>, E>;

template <typename E>
inline std::enable_if_t<is_gcontainer<std::decay_t<E>>::value, E> eval(E&& e)
{
  return std::forward<E>(e);
}

template <typename E>
inline std::enable_if_t<!is_gcontainer<std::decay_t<E>>::value,
                        gtensor<expr_value_type<E>, expr_dimension<E>(), expr_space_type<E>>>
eval(E&& e)
{
  return {std::forward<E>(e)};
}

} // namespace gt

#endif
