#ifndef GTENSOR_MEMORY_POOL_H
#define GTENSOR_MEMORY_POOL_H

#ifdef GTENSOR_USE_MEMORY_POOL

#ifdef GTENSOR_USE_RMM
#include <map>

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/host/pinned_memory_resource.hpp>
#elif GTENSOR_USE_UMPIRE
#include <umpire/Allocator.hpp>
#include <umpire/ResourceManager.hpp>
#include <umpire/strategy/DynamicPoolList.hpp>
#include <umpire/strategy/MixedPool.hpp>
#endif

#include "gtensor/defs.h"

namespace gt
{
namespace memory_pool
{

enum class memory_type
{
  device,
  managed,
  host_pinned
};

#if GTENSOR_USE_UMPIRE

#ifndef GTENSOR_UMPIRE_STRATEGY
#define GTENSOR_UMPIRE_STRATEGY DynamicPoolList
#endif

#define QUALIFY_UMPIRE_STRATEGY(x) umpire::strategy::x

class memory_pool
{
public:
  // using strategy = umpire::strategy::MixedPool;
  // using strategy = umpire::strategy::DynamicPoolList;
  using strategy = QUALIFY_UMPIRE_STRATEGY(GTENSOR_UMPIRE_STRATEGY);
  memory_pool()
    : rm_{umpire::ResourceManager::getInstance()},
      a_host_{
        rm_.makeAllocator<strategy>("PINNED_pool", rm_.getAllocator("PINNED"))},
      a_device_{
        rm_.makeAllocator<strategy>("DEVICE_pool", rm_.getAllocator("DEVICE"))},
      a_managed_{rm_.makeAllocator<strategy>("UM_pool", rm_.getAllocator("UM"))}
  {}

  template <memory_type MemType>
  void* allocate(size_type nbytes);

  template <memory_type MemType>
  void deallocate(void* p);

private:
  umpire::ResourceManager& rm_;
  umpire::Allocator a_host_;
  umpire::Allocator a_device_;
  umpire::Allocator a_managed_;
};

template <>
inline void* memory_pool::allocate<memory_type::host_pinned>(size_type nbytes)
{
  void* p = a_host_.allocate(nbytes);
  if (p == nullptr) {
    throw std::runtime_error("host allocate failed");
  }
  return p;
}

template <>
inline void memory_pool::deallocate<memory_type::host_pinned>(void* p)
{
  assert(p != nullptr);
  a_host_.deallocate(p);
}

template <>
inline void* memory_pool::allocate<memory_type::device>(size_type nbytes)
{
  void* p = a_device_.allocate(nbytes);
  if (p == nullptr) {
    throw std::runtime_error("device allocate failed");
  }
  return p;
}

template <>
inline void memory_pool::deallocate<memory_type::device>(void* p)
{
  assert(p != nullptr);
  a_device_.deallocate(p);
}

template <>
inline void* memory_pool::allocate<memory_type::managed>(size_type nbytes)
{
  void* p = a_managed_.allocate(nbytes);
  if (p == nullptr) {
    throw std::runtime_error("managed allocate failed");
  }
  return p;
}

template <>
inline void memory_pool::deallocate<memory_type::managed>(void* p)
{
  assert(p != nullptr);
  a_managed_.deallocate(p);
}

#undef QUALIFY_UMPIRE_STRATEGY

#elif GTENSOR_USE_RMM

class memory_pool
{
public:
  memory_pool()
    : mr_host_{},
      mr_cuda_device_{},
      mr_cuda_managed_{},
      mr_device_{&mr_cuda_device_},
      mr_managed_{&mr_cuda_managed_},
      allocated_host_{},
      allocated_device_{},
      allocated_managed_{}
  {}

  template <memory_type MemType>
  void* allocate(size_type nbytes);

  template <memory_type MemType>
  void deallocate(void* p);

private:
  rmm::mr::pinned_memory_resource mr_host_;
  rmm::mr::cuda_memory_resource mr_cuda_device_;
  rmm::mr::managed_memory_resource mr_cuda_managed_;
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> mr_device_;
  rmm::mr::pool_memory_resource<rmm::mr::managed_memory_resource> mr_managed_;

  std::map<void*, size_type> allocated_host_;
  std::map<void*, size_type> allocated_device_;
  std::map<void*, size_type> allocated_managed_;
};

template <>
inline void* memory_pool::allocate<memory_type::host_pinned>(size_type nbytes)
{
  void* p = mr_host_.allocate(nbytes);
  if (p == nullptr) {
    throw std::runtime_error("host allocate failed");
  }
  allocated_host_.emplace(std::make_pair(p, nbytes));
  return p;
}

template <>
inline void memory_pool::deallocate<memory_type::host_pinned>(void* p)
{
  assert(p != nullptr);
  auto it = allocated_host_.find(p);
  assert(it != allocated_host_.end());
  mr_host_.deallocate(p, it->second);
  allocated_host_.erase(it);
}

template <>
inline void* memory_pool::allocate<memory_type::device>(size_type nbytes)
{
  void* p = mr_device_.allocate(nbytes);
  if (p == nullptr) {
    throw std::runtime_error("device allocate failed");
  }
  allocated_device_.emplace(std::make_pair(p, nbytes));
  return p;
}

template <>
inline void memory_pool::deallocate<memory_type::device>(void* p)
{
  assert(p != nullptr);
  auto it = allocated_device_.find(p);
  assert(it != allocated_device_.end());
  mr_device_.deallocate(p, it->second);
  allocated_device_.erase(it);
}

template <>
inline void* memory_pool::allocate<memory_type::managed>(size_type nbytes)
{
  void* p = mr_managed_.allocate(nbytes);
  if (p == nullptr) {
    throw std::runtime_error("managed allocate failed");
  }
  allocated_managed_.emplace(std::make_pair(p, nbytes));
  return p;
}

template <>
inline void memory_pool::deallocate<memory_type::managed>(void* p)
{
  assert(p != nullptr);
  auto it = allocated_managed_.find(p);
  assert(it != allocated_managed_.end());
  mr_managed_.deallocate(p, it->second);
  allocated_managed_.erase(it);
}

#else

#error "No memory pool implementation specified"

#endif

inline memory_pool& get_instance()
{
  static memory_pool pool;
  return pool;
}

} // namespace memory_pool
} // namespace gt

#endif // GTENSOR_USE_MEMORY_POOL

#endif // GTENSOR_MEMORY_POOL_H
