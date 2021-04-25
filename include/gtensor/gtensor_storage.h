#ifndef GTENSOR_DEVICE_STORAGE_H
#define GTENSOR_DEVICE_STORAGE_H

#include <memory>
#include <type_traits>

#include "device_backend.h"
#include "device_copy.h"

namespace gt
{

namespace backend
{

/*! A container implementing the 'storage' API for gtensor in device memory.
 * Note that this is a small subset of the features in thrust::device_vector.
 * In particular, iterators are not yet supported.
 */
template <typename T, typename A, typename S>
class gtensor_storage
{
public:
  using value_type = T;
  using allocator_type = A;
  using pointer = typename std::allocator_traits<A>::pointer;
  using const_pointer = typename std::allocator_traits<A>::const_pointer;
  using reference = value_type&;
  using const_reference = const value_type&;
  using size_type = typename std::allocator_traits<A>::size_type;
  using space_type = S;

  gtensor_storage(size_type count)
    : data_(nullptr), size_(count), capacity_(count)
  {
    if (capacity_ > 0) {
      data_ = allocator_.allocate(capacity_);
    }
  }
  gtensor_storage() : gtensor_storage(0) {}

  ~gtensor_storage() { allocator_.deallocate(data_, capacity_); }

  // copy and move constructors
  gtensor_storage(const gtensor_storage& dv)
    : data_(nullptr), size_(0), capacity_(0)
  {
    resize_discard(dv.size_);

    if (size_ > 0) {
      copy(dv.data_, data_, size_);
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
      copy(dv.data_, data_, size_);
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
  void copy(const_pointer src, pointer dst, size_type count);
  void resize(size_type new_size, bool discard);
  void resize_discard(size_type new_size);

  pointer data_;
  size_type size_;
  size_type capacity_;
  allocator_type allocator_;
};

#ifdef GTENSOR_HAVE_DEVICE

template <typename T, typename A = standard::device_allocator<T>>
using device_storage = gtensor_storage<T, A, space::device>;

#endif

template <typename T, typename A = standard::host_allocator<T>>
using host_storage = gtensor_storage<T, A, space::host>;

template <typename T, typename A, typename O>
inline void gtensor_storage<T, A, O>::copy(const_pointer src, pointer dst,
                                           size_type count)
{
  backend::standard::copy<space_type, space_type>(src, dst, count);
}

template <typename T, typename A, typename O>
inline void gtensor_storage<T, A, O>::resize(
  gtensor_storage::size_type new_size, bool discard)
{
  if (capacity_ == 0) {
    if (new_size == 0) {
      return;
    }
    capacity_ = size_ = new_size;
    data_ = allocator_.allocate(capacity_);
  } else if (new_size > capacity_) {
    pointer new_data = allocator_.allocate(new_size);
    if (!discard && size_ > 0) {
      size_type copy_size = std::min(size_, new_size);
      copy(data_, new_data, copy_size);
    }
    allocator_.deallocate(data_, capacity_);
    data_ = new_data;
    capacity_ = size_ = new_size;
  } else {
    // TODO: set reallocate shrink threshold?
    size_ = new_size;
  }
}

template <typename T, typename A, typename O>
inline void gtensor_storage<T, A, O>::resize_discard(
  gtensor_storage::size_type new_size)
{
  resize(new_size, true);
}

template <typename T, typename A, typename O>
inline void gtensor_storage<T, A, O>::resize(
  gtensor_storage::size_type new_size)
{
  resize(new_size, false);
}

// ===================================================================
// equality operators (for testing)

template <typename T>
host_storage<T>& host_mirror(host_storage<T>& h)
{
  return h;
}

template <typename T>
const host_storage<T>& host_mirror(const host_storage<T>& h)
{
  return h;
}

template <typename T>
void copy(const host_storage<T>& d, const host_storage<T>& h)
{
  // this copy is, at this time, only for use with host_mirror,
  // which return a reference to the very same object in the host
  // case
  assert(&h == &d);
}

#ifdef GTENSOR_HAVE_DEVICE

template <typename T>
host_storage<T> host_mirror(const device_storage<T>& d)
{
  return host_storage<T>(d.size());
}

template <typename T>
void copy(const device_storage<T>& d, host_storage<T>& h)
{
  assert(h.size() == d.size());
  gt::backend::standard::copy<space::device, space::host>(d.data(), h.data(),
                                                          d.size());
}

template <typename T>
void copy(const host_storage<T>& h, device_storage<T>& d)
{
  assert(h.size() == d.size());
  gt::backend::standard::copy<space::host, space::device>(h.data(), d.data(),
                                                          h.size());
}

#endif

template <typename T, typename A, typename O>
bool operator==(const gtensor_storage<T, A, O>& v1,
                const gtensor_storage<T, A, O>& v2)
{
  if (v1.size() != v2.size()) {
    return false;
  }
  auto&& h1 = host_mirror(v1);
  auto&& h2 = host_mirror(v2);
  copy(v1, h1);
  copy(v2, h2);
  for (int i = 0; i < h1.size(); i++) {
    if (h1[i] != h2[i]) {
      return false;
    }
  }
  return true;
}

template <typename T, typename A, typename O>
bool operator!=(const gtensor_storage<T, A, O>& v1,
                const gtensor_storage<T, A, O>& v2)
{
  return !(v1 == v2);
}

} // end namespace backend

} // namespace gt

#endif // GTENSOR_DEVICE_STORAGE_H
