#ifndef GTENSOR_DEVICE_PTR_H
#define GTENSOR_DEVICE_PTR_H

#include "defs.h"
#include "helper.h"
#include "space_forward.h"

namespace gt
{
template <typename T>
class device_ptr
{
public:
  using self_type = device_ptr<T>;

  using element_type = T;
  using difference_type = std::ptrdiff_t;
  using space_type = gt::space::device;

  template <typename U>
  using rebind = device_ptr<U>;

  device_ptr() = default;
  device_ptr(std::nullptr_t){};
  template <
    typename U,
    std::enable_if_t<is_allowed_element_type_conversion<U, T>::value, int> = 0>
  explicit GT_INLINE device_ptr(U* p) : p_(p)
  {}
  template <
    typename U,
    std::enable_if_t<is_allowed_element_type_conversion<U, T>::value, int> = 0>
  GT_INLINE device_ptr(device_ptr<U> other) : p_(other.get())
  {}

  GT_INLINE T* get() const { return p_; }
  GT_INLINE T& operator*() const { return *p_; }
  GT_INLINE T& operator[](size_type i) const { return p_[i]; }
  GT_INLINE T* operator->() const { return p_; }

  explicit GT_INLINE operator bool() const { return bool(p_); }
  GT_INLINE self_type operator+(size_type off) const
  {
    return self_type(get() + off);
  }
  GT_INLINE self_type operator-(size_type off) const
  {
    return self_type(get() - off);
  }

  GT_INLINE difference_type operator-(const self_type& other) const
  {
    return get() - other.get();
  }

  template <
    typename U,
    std::enable_if_t<is_allowed_element_type_conversion<T, U>::value, int> = 0>
  GT_INLINE bool operator<(const device_ptr<U>& other) const
  {
    return get() < other.get();
  }
  template <
    typename U,
    std::enable_if_t<is_allowed_element_type_conversion<T, U>::value, int> = 0>
  GT_INLINE bool operator==(const device_ptr<U>& other) const
  {
    return get() == other.get();
  }
  template <
    typename U,
    std::enable_if_t<is_allowed_element_type_conversion<T, U>::value, int> = 0>
  GT_INLINE bool operator!=(const device_ptr<U>& other) const
  {
    return !(*this == other);
  }

  GT_INLINE bool operator==(std::nullptr_t) const { return get() == nullptr; }
  GT_INLINE bool operator!=(std::nullptr_t) const
  {
    return !(*this == nullptr);
  }

private:
  T* p_ = nullptr;
};

} // namespace gt

#endif
