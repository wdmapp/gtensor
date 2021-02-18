
#ifndef GTENSOR_SPACE_H
#define GTENSOR_SPACE_H

#include "defs.h"
#include "gtensor_storage.h"
#include "helper.h"
#include "meta.h"
#include "span.h"

#include <map>
#include <vector>

#ifdef GTENSOR_USE_THRUST
#include <thrust/device_allocator.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#endif

namespace gt
{

// ======================================================================
// caching_allocator

template <class T, class A>
struct caching_allocator : A
{
  using base_type = A;
  using value_type = typename A::value_type;
  using reference = typename A::reference;
  using const_reference = typename A::const_reference;
  using pointer = typename A::pointer;
  using const_pointer = typename A::const_pointer;
  using size_type = typename A::size_type;
  using difference_type = typename A::difference_type;

  caching_allocator() {}
  caching_allocator(const caching_allocator&) {}

  ~caching_allocator() {}

  pointer allocate(size_type cnt)
  {
    pointer p;
    auto it = free_.find(cnt);
    if (it != free_.end()) {
      p = it->second;
      free_.erase(it);
#ifdef DEBUG
      std::cout << "ALLOC: allocating " << cnt << " bytes from cache\n";
#endif
    } else {
#ifdef DEBUG
      std::cout << "ALLOC: total used " << used_ * sizeof(T) << ", allocing "
                << cnt * sizeof(T) << "\n";
#endif
      p = base_type::allocate(cnt);
      used_ += cnt;
#ifdef DEBUG
      std::cout << "ALLOC: allocating " << cnt << " bytes\n";
#endif
    }
    allocated_.emplace(std::make_pair(p, cnt));
    return p;
  }

  void deallocate(pointer p, size_type cnt)
  {
    gt::synchronize();
    auto it = allocated_.find(p);
    assert(it != allocated_.end());
    free_.emplace(std::make_pair(it->second, p));
    allocated_.erase(it);
#ifdef DEBUG
    std::cout << "ALLOC: deallocing cnt " << cnt
              << " #allocated = " << allocated_.size()
              << " #free = " << free_.size() << "\n";
#endif
  }

  GT_INLINE void construct(pointer) {}

  static void clear_cache()
  {
    for (auto it = free_.begin(); it != free_.end(); it++) {
      base_type::deallocate(it->second, it->first);
    }
    free_.clear();
  }

  template <class U>
  struct rebind
  {
    using other = caching_allocator<U, typename A::template rebind<U>::other>;
  };

private:
  static std::multimap<size_type, pointer> free_;
  static std::map<pointer, size_type> allocated_;
  static size_t used_;
};

template <class T, class A>
std::multimap<typename caching_allocator<T, A>::size_type,
              typename caching_allocator<T, A>::pointer>
  caching_allocator<T, A>::free_;

template <class T, class A>
std::map<typename caching_allocator<T, A>::pointer,
         typename caching_allocator<T, A>::size_type>
  caching_allocator<T, A>::allocated_;

template <class T, class A>
size_t caching_allocator<T, A>::used_;

template <class T, class AT, class U, class AU>
inline bool operator==(const caching_allocator<T, AT>&,
                       const caching_allocator<U, AU>&)
{
  return std::is_same<AT, AU>::value;
}

template <class T, class AT, class U, class AU>
inline bool operator!=(const caching_allocator<T, AT>& a,
                       const caching_allocator<U, AU>& b)
{
  return !(a == b);
}

// ======================================================================
// space

namespace space
{

struct any;

struct kernel;

struct host
{
  template <typename T>
#ifdef GTENSOR_USE_THRUST
  using Vector = thrust::host_vector<T>;
#else
  using Vector = gt::backend::host_storage<T>;
#endif
  template <typename T>
  using Span = span<T>;
};

#ifdef GTENSOR_HAVE_DEVICE

#ifdef GTENSOR_USE_THRUST

#if GTENSOR_DEVICE_CUDA && THRUST_VERSION <= 100903
template <typename T>
using device_allocator =
  caching_allocator<T, thrust::device_malloc_allocator<T>>;
#else
template <typename T>
using device_allocator = caching_allocator<T, thrust::device_allocator<T>>;
#endif

struct device
{
  template <typename T>
  using Vector = thrust::device_vector<T, device_allocator<T>>;
  template <typename T>
  using Span = device_span<T>;
};

#else

struct device
{
  template <typename T>
  using Vector = gt::backend::device_storage<T>;
  template <typename T>
  using Span = device_span<T>;
};

#endif // GTENSOR_USE_THRUST

#else // not GTENSOR_HAVE_DEVICE

using device = host;

#endif

} // namespace space

// ======================================================================
// has_space_type

template <typename E, typename S, typename = void>
struct has_space_type : std::false_type
{};

template <typename T, typename S>
struct has_space_type<T, S,
                      gt::meta::void_t<std::enable_if_t<
                        std::is_same<expr_space_type<T>, S>::value>>>
  : std::true_type
{};

template <typename T, typename S>
constexpr bool has_space_type_v = has_space_type<T, S>::value;

// ======================================================================
// has_space_type_device

template <typename E, typename = void>
struct has_space_type_device : has_space_type<E, gt::space::device>
{};

template <typename T>
constexpr bool has_space_type_device_v = has_space_type_device<T>::value;

// ======================================================================
// has_space_type_host

template <typename E, typename = void>
struct has_space_type_host : has_space_type<E, gt::space::host>
{};

template <typename T>
constexpr bool has_space_type_host_v = has_space_type_host<T>::value;

} // namespace gt

#endif
