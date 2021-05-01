
#ifndef GTENSOR_SPAN_H
#define GTENSOR_SPAN_H

#include <cassert>

#include "defs.h"
#include "macros.h"
#include "pointer_traits.h"

#ifdef GTENSOR_HAVE_DEVICE
#include "gtensor_storage.h"
#endif

namespace gt
{

// ======================================================================
// span
//
// very minimal, just enough to support making a gtensor_span
// Note that the span has pointer semantics, in that copying does
// not copy the underlying data, just the pointer and size, and
// requesting access to the underlying data from a const instance
// via data() and operator[] returns a non-const reference allowing
// modification. This is consistent with the C++20 standardized
// span and with gsl::span. To not allow modification, the underlying
// data type can be const.
//
// This is not a plug-in replacement with std::span. It does only support
// dynamic_extent, and is otherwise missing features, but it does support
// custom pointer and reference types, as needed to handle device_pointer

template <typename T, typename Ptr = T*>
class span
{
public:
  using element_type = T;
  using value_type = std::remove_cv_t<T>;

  using pointer = typename pointer_traits<Ptr>::pointer;
  using const_pointer = typename pointer_traits<Ptr>::const_pointer;
  using reference = typename pointer_traits<Ptr>::reference;
  using const_reference = typename pointer_traits<Ptr>::const_reference;
  using iterator = pointer;
  using size_type = gt::size_type;

  span() = default;
  GT_INLINE span(pointer data, size_type size) : data_{data}, size_{size} {}

  span(const span& other) = default;

  template <class OtherT,
            std::enable_if_t<
              is_allowed_element_type_conversion<OtherT, T>::value, int> = 0>
  GT_INLINE span(
    const span<OtherT, typename pointer_traits<Ptr>::template rebind<OtherT>>&
      other)
    : data_{other.data()}, size_{other.size()}
  {}

  span& operator=(const span& other) = default;

  GT_INLINE pointer data() const { return data_; }
  GT_INLINE size_type size() const { return size_; }

  GT_INLINE iterator begin() const { return data_; }
  GT_INLINE iterator end() const { return data_ + size_; }

  GT_INLINE reference operator[](size_type i) const { return data_[i]; }

private:
  pointer data_ = nullptr;
  size_type size_ = 0;
};

#ifdef GTENSOR_HAVE_DEVICE

#endif // GTENSOR_HAVE_DEVICE

} // namespace gt

#endif // GTENSOR_SPAN_H
