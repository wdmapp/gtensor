
#ifndef GTENSOR_GTENSOR_VIEW_H
#define GTENSOR_GTENSOR_VIEW_H

#include "device_backend.h"
#include "gcontainer.h"

namespace gt
{

// ======================================================================
// gtensor_view

template <typename T, int N, typename S = space::host>
class gtensor_view;

template <typename T, int N, typename S>
struct gtensor_inner_types<gtensor_view<T, N, S>>
{
  using space_type = S;
  constexpr static size_type dimension = N;

  using storage_type = typename space_type::template Span<T>;
  using value_type = typename storage_type::value_type;
  using pointer = typename storage_type::pointer;
  using const_pointer = typename storage_type::const_pointer;
  using reference = typename storage_type::reference;
  using const_reference = typename storage_type::const_reference;
};

template <typename T, int N, typename S>
class gtensor_view : public gstrided<gtensor_view<T, N, S>>
{
public:
  using self_type = gtensor_view<T, N, S>;
  using base_type = gstrided<self_type>;
  using inner_types = gtensor_inner_types<self_type>;
  using storage_type = typename inner_types::storage_type;

  using value_type = typename storage_type::value_type;
  using pointer = typename storage_type::pointer;
  using const_pointer = typename storage_type::const_pointer;
  using reference = typename storage_type::reference;
  using const_reference = typename storage_type::const_reference;

  using base_type::dimension;
  using typename base_type::shape_type;
  using typename base_type::strides_type;

  gtensor_view() = default;
  gtensor_view(pointer data, const shape_type& shape,
               const strides_type& strides);

  template <typename E>
  self_type& operator=(const expression<E>& e);

  // FIXME, const correctness
  gtensor_view to_kernel() const;

  GT_INLINE const_pointer data() const;
  GT_INLINE pointer data();

private:
  GT_INLINE const_reference data_access_impl(size_type i) const;
  GT_INLINE reference data_access_impl(size_type i);

  storage_type storage_;

  friend class gstrided<self_type>;
  friend class gcontainer<self_type>;
};

// ======================================================================
// gtensor_view implementation

template <typename T, int N, typename S>
inline gtensor_view<T, N, S>::gtensor_view(pointer data,
                                           const shape_type& shape,
                                           const strides_type& strides)
  : base_type(shape, strides), storage_(data, calc_size(shape))
{
#ifndef NDEBUG
#ifdef GTENSOR_DEVICE_CUDA
  if (std::is_same<S, space::device>::value) {
    cudaPointerAttributes attr;
    gtGpuCheck(
      cudaPointerGetAttributes(&attr, gt::backend::raw_pointer_cast(data)));
    assert(attr.type == cudaMemoryTypeDevice ||
           attr.type == cudaMemoryTypeManaged);
  }
#elif defined(GTENSOR_DEVICE_HIP)
  if (std::is_same<S, space::device>::value) {
    hipPointerAttribute_t attr;
    hipCheck(
      hipPointerGetAttributes(&attr, gt::backend::raw_pointer_cast(data)));
    assert(attr.memoryType == hipMemoryTypeDevice || attr.isManaged);
  }
#endif
#endif
}

template <typename T, int N, typename S>
template <typename E>
inline auto gtensor_view<T, N, S>::operator=(const expression<E>& e)
  -> self_type&
{
  assign(*this, e.derived());
  return *this;
}

template <typename T, int N, typename S>
inline auto gtensor_view<T, N, S>::to_kernel() const -> gtensor_view
{
  return *this;
}

template <typename T, int N, typename S>
GT_INLINE auto gtensor_view<T, N, S>::data() const -> const_pointer
{
  return storage_.data();
}

template <typename T, int N, typename S>
GT_INLINE auto gtensor_view<T, N, S>::data() -> pointer
{
  return storage_.data();
}

template <typename T, int N, typename S>
GT_INLINE auto gtensor_view<T, N, S>::data_access_impl(size_t i) const
  -> const_reference
{
  return storage_[i];
}

template <typename T, int N, typename S>
GT_INLINE auto gtensor_view<T, N, S>::data_access_impl(size_t i) -> reference
{
  return storage_[i];
}

// ======================================================================
// adapt

template <size_type N, typename T>
gtensor_view<T, N> adapt(T* data, const shape_type<N>& shape)
{
  return gtensor_view<T, N>(data, shape, calc_strides(shape));
}

template <size_type N, typename T>
gtensor_view<T, N> adapt(T* data, const int* shape_data)
{
  return adapt<N, T>(data, {shape_data, N});
}

#ifdef GTENSOR_HAVE_DEVICE
template <size_type N, typename T>
gtensor_view<T, N, space::device> adapt_device(T* data,
                                               const shape_type<N>& shape)
{
  return gtensor_view<T, N, space::device>(
    gt::backend::device_pointer_cast(data), shape, calc_strides(shape));
}

template <size_type N, typename T>
gtensor_view<T, N, space::device> adapt_device(T* data, const int* shape_data)
{
  return adapt_device<N, T>(data, {shape_data, N});
}
#endif

} // namespace gt

#endif
