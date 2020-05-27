#ifndef GTENSOR_DEVICE_STORAGE_H
#define GTENSOR_DEVICE_STORAGE_H

#include <type_traits>

#include "device_backend.h"

namespace gt {

namespace backend {

/*! A container implementing the 'storage' API for gtensor in device memory.
 * Note that this is a small subset of the features in thrust::device_vector.
 * In particular, iterators are not yet supported.
 */
template <typename T, typename Allocator>
class gtensor_storage {
  public:
    using value_type = T;
    using allocator_type = Allocator;

    using pointer = typename std::add_pointer<value_type>::type;
    using const_pointer = typename std::add_const<pointer>::type;
    using reference = typename std::add_lvalue_reference<value_type>::type;
    using const_reference = typename std::add_const<reference>::type;
    using size_type = gt::size_type;

    gtensor_storage(size_type count)
    : data_(nullptr), size_(count), capacity_(count) {
      if (capacity_ > 0) {
        data_ = allocator_type::allocate(capacity_);
      }
    }
    gtensor_storage() : gtensor_storage(0) {}

    ~gtensor_storage() {
      if (data_ != nullptr) {
        allocator_type::deallocate(data_);
      }
    }

    // copy and move constructors
    gtensor_storage(const gtensor_storage &dv) 
    : data_(nullptr), size_(0), capacity_(0) {
      resize(dv.size());

      if (dv.size() > 0) {
        device_memcpy_dd(data(), dv.data(), size()*sizeof(value_type));
      }
    }

    gtensor_storage(gtensor_storage &&dv) = delete;

    // operators
    reference operator[](size_type i);
    const_reference operator[](size_type i) const;

    gtensor_storage& operator=(const gtensor_storage &dv) {
      resize(dv.size());

      if (dv.size() > 0) {
        device_memcpy_dd(data(), dv.data(), size()*sizeof(value_type));
      }

      return *this;
    }

    gtensor_storage& operator=(gtensor_storage &&dv) = delete;

    // functions
    void resize(size_type new_size);
    size_type size() const;
    pointer data();
    const_pointer data() const;

  private:
    value_type *data_;
    size_type size_;
    size_type capacity_;
};

#ifdef GTENSOR_HAVE_DEVICE
template <typename T>
using device_storage = gtensor_storage<T, device_allocator<T>>;

template <typename T>
using host_storage = gtensor_storage<T, host_allocator<T>>;
#endif


template <typename T, typename A>
inline void gtensor_storage<T, A>::resize(gtensor_storage::size_type new_size) {
  if (new_size == 0) {
    if (data_ != nullptr)
      allocator_type::deallocate(data_);
    capacity_ = size_ = 0;
  } else if (capacity_ == 0) {
    capacity_ = size_ = new_size;
    data_ = allocator_type::allocate(capacity_);
  } else if (new_size > capacity_) {
    if (data_ != nullptr)
      allocator_type::deallocate(data_);
    capacity_ = size_ = new_size;
    data_ = allocator_type::allocate(capacity_);
  } else if (new_size < capacity_) {
    // TODO: set shrink threshold?
    size_ = new_size;
  }
}


template <typename T, typename A>
inline typename gtensor_storage<T, A>::reference
gtensor_storage<T, A>::operator[](gtensor_storage::size_type i) {
  return data_[i];
}


template <typename T, typename A>
inline typename gtensor_storage<T, A>::const_reference
gtensor_storage<T, A>::operator[](gtensor_storage::size_type i) const {
  return data_[i];
}


template <typename T, typename A>
inline typename gtensor_storage<T, A>::size_type
gtensor_storage<T, A>::size() const {
  return size_;
}


template <typename T, typename A>
inline typename gtensor_storage<T, A>::pointer
gtensor_storage<T, A>::data() {
  return data_;
}


template <typename T, typename A>
inline typename gtensor_storage<T, A>::const_pointer
gtensor_storage<T, A>::data() const {
  return data_;
}

} // end namespace backend

} // end namespace gt

#endif // GTENSOR_DEVICE_STORAGE_H
