
#ifndef GTENSOR_SPACE_H
#define GTENSOR_SPACE_H

#include "defs.h"
#include "span.h"

#include <map>
#include <vector>

#ifdef __CUDACC__
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

  using base_type::base_type;

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
      p = base_type::allocate(cnt);
#ifdef DEBUG
      std::cout << "ALLOC: allocating " << cnt << " bytes\n";
#endif
    }
    allocated_.emplace(std::make_pair(p, cnt));
    return p;
  }

  void deallocate(pointer p, size_type cnt)
  {
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

  void clear_cache()
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
};

template <class T, class A>
std::multimap<typename caching_allocator<T, A>::size_type,
              typename caching_allocator<T, A>::pointer>
  caching_allocator<T, A>::free_;

template <class T, class A>
std::map<typename caching_allocator<T, A>::pointer,
         typename caching_allocator<T, A>::size_type>
  caching_allocator<T, A>::allocated_;

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
#ifdef __CUDACC__
  using Vector = thrust::host_vector<T>;
#else
  using Vector = std::vector<T>;
#endif
  template <typename T>
  using Span = span<T>;
};

#ifdef __CUDACC__

#if THRUST_VERSION <= 100903
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

using device = host;

#endif

} // namespace space

} // namespace gt

#endif
