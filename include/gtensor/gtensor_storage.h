#ifndef GTENSOR_DEVICE_STORAGE_H
#define GTENSOR_DEVICE_STORAGE_H

#include <type_traits>

#include "device_backend.h"

namespace gt
{

namespace backend
{

/*! A container implementing the 'storage' API for gtensor in device memory.
 * Note that this is a small subset of the features in thrust::device_vector.
 * In particular, iterators are not yet supported.
 */
template <typename T, typename Allocator>
class gtensor_storage
{
public:
  using element_type = T;
  using value_type = std::remove_cv_t<T>;
  using allocator_type = Allocator;

  using pointer = std::add_pointer_t<element_type>;
  using const_pointer = std::add_pointer_t<std::add_const_t<element_type>>;
  using reference = std::add_lvalue_reference_t<element_type>;
  using const_reference =
    std::add_lvalue_reference_t<std::add_const_t<element_type>>;
  using size_type = gt::size_type;

  gtensor_storage(size_type count)
    : data_(nullptr), size_(count), capacity_(count)
  {
    if (capacity_ > 0) {
      data_ = allocator_type::allocate(capacity_);
    }
  }
  gtensor_storage() : gtensor_storage(0) {}

  ~gtensor_storage() { allocator_type::deallocate(data_); }

  // copy and move constructors
  gtensor_storage(const gtensor_storage& dv)
    : data_(nullptr), size_(0), capacity_(0)
  {
    resize_discard(dv.size_);

    if (size_ > 0) {
      allocator_type::copy(dv.data_, data_, size_);
    }
  }

  gtensor_storage(gtensor_storage&& dv)
    : data_(dv.data_), size_(dv.size_), capacity_(dv.capacity_)
  {
    dv.size_ = dv.capacity_ = 0;
    dv.data_ = nullptr;
  }

  // operators
  reference operator[](size_type i) { return data_[i]; }
  const_reference operator[](size_type i) const { return data_[i]; }

  gtensor_storage& operator=(const gtensor_storage& dv)
  {
    resize_discard(dv.size_);

    if (size_ > 0) {
      allocator_type::copy(dv.data_, data_, size_);
    }

    return *this;
  }

  gtensor_storage& operator=(gtensor_storage&& dv)
  {
    data_ = dv.data_;
    size_ = dv.size_;
    capacity_ = dv.capacity_;

    dv.size_ = dv.capacity_ = 0;
    dv.data_ = nullptr;

    return *this;
  }

  // functions
  void resize(size_type new_size);

  size_type size() const { return size_; }
  size_type capacity() const { return capacity_; }
  pointer data() { return data_; }
  const_pointer data() const { return data_; }

private:
  void resize(size_type new_size, bool discard);
  void resize_discard(size_type new_size);
  pointer data_;
  size_type size_;
  size_type capacity_;
};

#ifdef GTENSOR_HAVE_DEVICE
template <typename T>
using device_storage = gtensor_storage<T, device_allocator<T>>;
#endif

template <typename T>
using host_storage = gtensor_storage<T, host_allocator<T>>;

template <typename T, typename A>
inline void gtensor_storage<T, A>::resize(gtensor_storage::size_type new_size,
                                          bool discard)
{
  if (capacity_ == 0) {
    if (new_size == 0) {
      return;
    }
    capacity_ = size_ = new_size;
    data_ = allocator_type::allocate(capacity_);
  } else if (new_size > capacity_) {
    pointer new_data = allocator_type::allocate(new_size);
    if (!discard && size_ > 0) {
      size_type copy_size = std::min(size_, new_size);
      allocator_type::copy(data_, new_data, copy_size);
    }
    allocator_type::deallocate(data_);
    data_ = new_data;
    capacity_ = size_ = new_size;
  } else {
    // TODO: set reallocate shrink threshold?
    size_ = new_size;
  }
}

template <typename T, typename A>
inline void gtensor_storage<T, A>::resize_discard(
  gtensor_storage::size_type new_size)
{
  resize(new_size, true);
}

template <typename T, typename A>
inline void gtensor_storage<T, A>::resize(gtensor_storage::size_type new_size)
{
  resize(new_size, false);
}

// ===================================================================
// equality operators (for testing)

template <typename T>
bool operator==(const gtensor_storage<T, host_allocator<T>>& v1,
                const gtensor_storage<T, host_allocator<T>>& v2)
{
  if (v1.size() != v2.size()) {
    return false;
  }
  for (int i = 0; i < v1.size(); i++) {
    if (v1[i] != v2[i]) {
      return false;
    }
  }
  return true;
}

#ifdef GTENSOR_HAVE_DEVICE

template <typename T>
bool operator==(const gtensor_storage<T, device_allocator<T>>& v1,
                const gtensor_storage<T, device_allocator<T>>& v2)
{
  if (v1.size() != v2.size()) {
    return false;
  }
  host_storage<T> h1(v1.size());
  host_storage<T> h2(v2.size());
  device_copy_dh(v1.data(), h1.data(), v1.size());
  device_copy_dh(v2.data(), h2.data(), v2.size());
  for (int i = 0; i < v1.size(); i++) {
    if (h1[i] != h2[i]) {
      return false;
    }
  }
  return true;
}

#endif

template <typename T, typename A>
bool operator!=(const gtensor_storage<T, A>& v1,
                const gtensor_storage<T, A>& v2)
{
  return !(v1 == v2);
}

} // end namespace backend

} // end namespace gt

#endif // GTENSOR_DEVICE_STORAGE_H
