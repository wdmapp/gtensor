
#include <iostream>
#include <map>
#include <memory>

namespace gt
{
namespace allocator
{

// ======================================================================
// caching_allocator

template <class T, class A>
struct caching_allocator : A
{
  using base_type = A;
  using value_type = typename std::allocator_traits<A>::value_type;
  using pointer = typename std::allocator_traits<A>::pointer;
  using const_pointer = typename std::allocator_traits<A>::const_pointer;
  using size_type = typename std::allocator_traits<A>::size_type;
  using difference_type = typename std::allocator_traits<A>::difference_type;

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
    if (p) {
      allocated_.emplace(std::make_pair(p, cnt));
    }
    return p;
  }

  void deallocate(pointer p, size_type cnt)
  {
    gt::synchronize();
    if (p) {
      auto it = allocated_.find(p);
      assert(it != allocated_.end());
      free_.emplace(std::make_pair(it->second, p));
      allocated_.erase(it);
    }
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

} // namespace allocator
} // namespace gt
