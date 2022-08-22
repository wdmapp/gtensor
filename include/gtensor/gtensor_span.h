
#ifndef GTENSOR_GTENSOR_VIEW_H
#define GTENSOR_GTENSOR_VIEW_H

#include <cassert>
#include <sstream>
#include <type_traits>

#include "device_backend.h"
#include "macros.h"
#include "space.h"
#include "span.h"

namespace gt
{

// ======================================================================
// gtensor_span

template <typename T, size_type N, typename S = space::host>
class gtensor_span;

template <typename T, size_type N, typename S>
struct gtensor_inner_types<gtensor_span<T, N, S>>
{
  using space_type = S;
  constexpr static size_type dimension = N;

  using storage_type = typename gt::span<T, gt::space_pointer<T, S>>;
  using value_type = typename storage_type::value_type;
  using pointer = typename storage_type::pointer;
  using const_pointer = typename storage_type::const_pointer;
  using reference = typename storage_type::reference;
  using const_reference = typename storage_type::const_reference;
};

template <typename T, size_type N, typename S>
class gtensor_span : public gstrided<gtensor_span<T, N, S>>
{
public:
  using self_type = gtensor_span<T, N, S>;
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

  gtensor_span() = default;
  GT_INLINE gtensor_span(pointer data, const shape_type& shape,
                         const strides_type& strides);

  gtensor_span(const gtensor_span& other) = default;

  // Allow automatic conversion to const element_type
  template <class OtherT,
            std::enable_if_t<
              is_allowed_element_type_conversion<OtherT, T>::value, int> = 0>
  gtensor_span(const gtensor_span<OtherT, N, S>& other)
    : base_type{other.shape(), other.strides()},
      storage_{other.data(), other.size()}
  {}

  // Implicit conversion from a gtensor object with the same or compatible
  // element type
  template <
    class EC,
    std::enable_if_t<
      is_allowed_element_type_conversion<typename EC::value_type, T>::value &&
        std::is_same<S, typename space::storage_traits<EC>::space_type>::value,
      int> = 0>
  gtensor_span(gtensor_container<EC, N>& other)
    : base_type{other.shape(), other.strides()},
      storage_{other.data(), other.size()}
  {}

  template <
    class EC,
    std::enable_if_t<
      is_allowed_element_type_conversion<typename EC::value_type, T>::value &&
        std::is_same<S, typename space::storage_traits<EC>::space_type>::value,
      int> = 0>
  gtensor_span(const gtensor_container<EC, N>& other)
    : base_type{other.shape(), other.strides()},
      storage_{other.data(), other.size()}
  {}

  gtensor_span& operator=(const gtensor_span& other) = default;

  template <typename E>
  self_type& operator=(const expression<E>& e);

  void fill(const value_type v);

  gtensor_span to_kernel() const;

  GT_INLINE pointer data() const;

  template <typename... Args>
  GT_INLINE reference operator()(Args&&... args) const;

  GT_INLINE reference operator[](const shape_type& idx) const;

  GT_INLINE reference data_access(size_type i) const;

  inline std::string typestr() const&;

private:
  storage_type storage_;

  friend class gstrided<self_type>;

  template <typename Idx, size_type... I>
  GT_INLINE reference access(std::index_sequence<I...>, const Idx& idx) const
  {
    return (*this)(idx[I]...);
  }
};

// ======================================================================
// gtensor_span implementation

template <typename T, size_type N, typename S>
GT_INLINE gtensor_span<T, N, S>::gtensor_span(pointer data,
                                              const shape_type& shape,
                                              const strides_type& strides)
  : base_type(shape, strides), storage_(data, calc_size(shape))
{
#if defined(GTENSOR_HAVE_DEVICE) && defined(GTENSOR_ADDRESS_CHECK)
  bool check_address = false;
#ifdef GTENSOR_DEVICE_SYCL
  // special case SYCL to handle the host backend, which says that
  // even device pointers are not device addresses (true from a hardware
  // perspective, even if it's logically false in gtensor).
  sycl::device d = gt::backend::sycl::get_queue().get_device();
  if ((d.is_gpu() || d.is_cpu()) && std::is_same<S, space::device>::value) {
    check_address = true;
  }
#else  // not SYCL
  if (std::is_same<S, space::device>::value) {
    check_address = true;
  }
#endif // GTENSOR_DEVICE_SYCL
  if (check_address) {
    gtGpuAssert(
      gt::backend::clib::is_device_accessible(gt::raw_pointer_cast(data)),
      "host address passed to device span");
  }
#endif
}

template <typename T, size_type N, typename S>
template <typename E>
inline auto gtensor_span<T, N, S>::operator=(const expression<E>& e)
  -> self_type&
{
  assign(*this, e.derived());
  return *this;
}

template <typename T, size_type N, typename S>
inline void gtensor_span<T, N, S>::fill(const value_type v)
{
  if (v == T(0)) {
    gt::fill(this->data(), this->data() + this->size(), 0);
  } else {
    assign(*this, scalar(v));
  }
}

template <typename T, size_type N, typename S>
inline auto gtensor_span<T, N, S>::to_kernel() const -> gtensor_span
{
  return *this;
}

template <typename T, size_type N, typename S>
GT_INLINE auto gtensor_span<T, N, S>::data() const -> pointer
{
  return storage_.data();
}

template <typename T, size_type N, typename S>
GT_INLINE auto gtensor_span<T, N, S>::data_access(size_t i) const -> reference
{
  return storage_[i];
}

template <typename T, size_type N, typename S>
template <typename... Args>
GT_INLINE auto gtensor_span<T, N, S>::operator()(Args&&... args) const
  -> reference
{
  return data_access(base_type::index(std::forward<Args>(args)...));
}

template <typename T, size_type N, typename S>
GT_INLINE auto gtensor_span<T, N, S>::operator[](const shape_type& idx) const
  -> reference
{
  return access(std::make_index_sequence<shape_type::dimension>(), idx);
}

template <typename T, size_type N, typename S>
inline std::string gtensor_span<T, N, S>::typestr() const&
{
  std::stringstream s;
  s << "s" << N << "<" << get_type_name<T>() << ">" << this->shape()
    << this->strides();
  return s.str();
}

// ======================================================================
// adapt

template <size_type N, typename T>
GT_INLINE gtensor_span<T, N> adapt(T* data, const shape_type<N>& shape)
{
  return gtensor_span<T, N>(data, shape, calc_strides(shape));
}

template <size_type N, typename T>
gtensor_span<T, N> adapt(T* data, const int* shape_data)
{
  return adapt<N, T>(data, {shape_data, N});
}

template <size_type N, typename T>
GT_INLINE gtensor_span<T, N, space::device> adapt_device(
  T* data, const shape_type<N>& shape)
{
  return gtensor_span<T, N, space::device>(gt::device_pointer_cast(data), shape,
                                           calc_strides(shape));
}

template <size_type N, typename T>
gtensor_span<T, N, space::device> adapt_device(T* data, const int* shape_data)
{
  return adapt_device<N, T>(data, {shape_data, N});
}

// ======================================================================
// is_gtensor_span

template <typename E>
struct is_gtensor_span : std::false_type
{};

template <typename T, size_type N, typename S>
struct is_gtensor_span<gtensor_span<T, N, S>> : std::true_type
{};

} // namespace gt

#endif
