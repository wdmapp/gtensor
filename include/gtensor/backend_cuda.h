
#ifndef GTENSOR_BACKEND_CUDA_H
#define GTENSOR_BACKEND_CUDA_H

#include "backend_common.h"

#include <cuda_runtime_api.h>

// #include "thrust/cuda/system/execution_policy.h"
#include "thrust/execution_policy.h"

#ifdef GTENSOR_USE_RMM
#include <map>

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/host/pinned_memory_resource.hpp>
#endif

// ======================================================================
// gt::backend::cuda

namespace gt
{
namespace backend
{

namespace allocator_impl
{

#ifdef GTENSOR_USE_RMM

class memory_resources
{
public:
  memory_resources()
    : mr_host_{},
      mr_cuda_device_{},
      mr_cuda_managed_{},
      mr_device_{&mr_cuda_device_},
      mr_managed_{&mr_cuda_managed_},
      allocated_host_{},
      allocated_device_{},
      allocated_managed_{}
  {}

  void* allocate_host(size_type nbytes)
  {
    void* p = mr_host_.allocate(nbytes);
    if (p == nullptr) {
      throw std::runtime_error("host allocate failed");
    }
    allocated_host_.emplace(std::make_pair(p, nbytes));
    return p;
  }

  void deallocate_host(void* p)
  {
    assert(p != nullptr);
    auto it = allocated_host_.find(p);
    assert(it != allocated_host_.end());
    mr_host_.deallocate(p, it->second);
    allocated_host_.erase(it);
  }

  void* allocate_device(size_type nbytes)
  {
    void* p = mr_device_.allocate(nbytes);
    if (p == nullptr) {
      throw std::runtime_error("device allocate failed");
    }
    allocated_device_.emplace(std::make_pair(p, nbytes));
    return p;
  }

  void deallocate_device(void* p)
  {
    assert(p != nullptr);
    auto it = allocated_device_.find(p);
    assert(it != allocated_device_.end());
    mr_device_.deallocate(p, it->second);
    allocated_device_.erase(it);
  }

  void* allocate_managed(size_type nbytes)
  {
    void* p = mr_managed_.allocate(nbytes);
    if (p == nullptr) {
      throw std::runtime_error("managed allocate failed");
    }
    allocated_managed_.emplace(std::make_pair(p, nbytes));
    return p;
  }

  void deallocate_managed(void* p)
  {
    assert(p != nullptr);
    auto it = allocated_managed_.find(p);
    assert(it != allocated_managed_.end());
    mr_managed_.deallocate(p, it->second);
    allocated_managed_.erase(it);
  }

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

inline memory_resources& rmm_get_mr_instance()
{
  static memory_resources mr;
  return mr;
}

template <>
struct gallocator<gt::space::cuda>
{
  template <typename T>
  static T* allocate(size_type n)
  {
    auto nbytes = sizeof(T) * n;
    return static_cast<T*>(rmm_get_mr_instance().allocate_device(nbytes));
  }

  template <typename T>
  static void deallocate(T* p)
  {
    rmm_get_mr_instance().deallocate_device(p);
  }
};

template <>
struct gallocator<gt::space::cuda_managed>
{
  template <typename T>
  static T* allocate(size_type n)
  {
    auto nbytes = sizeof(T) * n;
    auto mtype = gt::backend::get_managed_memory_type();
    if (mtype == gt::backend::managed_memory_type::managed) {
      return static_cast<T*>(rmm_get_mr_instance().allocate_managed(nbytes));
    } else if (mtype == gt::backend::managed_memory_type::device) {
      return static_cast<T*>(rmm_get_mr_instance().allocate_device(nbytes));
    } else {
      throw std::runtime_error("unsupported managed memory type for backend");
    }
  }

  template <typename T>
  static void deallocate(T* p)
  {
    auto mtype = gt::backend::get_managed_memory_type();
    if (mtype == gt::backend::managed_memory_type::managed) {
      rmm_get_mr_instance().deallocate_managed(p);
    } else if (mtype == gt::backend::managed_memory_type::device) {
      rmm_get_mr_instance().deallocate_device(p);
    } else {
      throw std::runtime_error("unsupported managed memory type for backend");
    }
  }
};

template <>
struct gallocator<gt::space::cuda_host>
{
  template <typename T>
  static T* allocate(size_type n)
  {
    auto nbytes = sizeof(T) * n;
    return static_cast<T*>(rmm_get_mr_instance().allocate_host(nbytes));
  }

  template <typename T>
  static void deallocate(T* p)
  {
    rmm_get_mr_instance().deallocate_host(p);
  }
};

#else // GTENSOR_USE_RMM

template <>
struct gallocator<gt::space::cuda>
{
  template <typename T>
  static T* allocate(size_type n)
  {
    T* p;
    gtGpuCheck(cudaMalloc(&p, sizeof(T) * n));
    return p;
  }

  template <typename T>
  static void deallocate(T* p)
  {
    gtGpuCheck(cudaFree(p));
  }
};

template <>
struct gallocator<gt::space::cuda_managed>
{
  template <typename T>
  static T* allocate(size_type n)
  {
    T* p;
    auto nbytes = sizeof(T) * n;
    auto mtype = gt::backend::get_managed_memory_type();
    if (mtype == gt::backend::managed_memory_type::managed) {
      gtGpuCheck(cudaMallocManaged(&p, nbytes));
    } else if (mtype == gt::backend::managed_memory_type::device) {
      gtGpuCheck(cudaMalloc(&p, nbytes));
    } else {
      throw std::runtime_error("unsupported managed memory type for backend");
    }
    return p;
  }

  template <typename T>
  static void deallocate(T* p)
  {
    gtGpuCheck(cudaFree(p));
  }
};

template <>
struct gallocator<gt::space::cuda_host>
{
  template <typename T>
  static T* allocate(size_type n)
  {
    T* p;
    gtGpuCheck(cudaMallocHost(&p, sizeof(T) * n));
    return p;
  }

  template <typename T>
  static void deallocate(T* p)
  {
    gtGpuCheck(cudaFreeHost(p));
  }
};

#endif // GTENSOR_USE_RMM

} // namespace allocator_impl

namespace copy_impl
{

template <typename InputPtr, typename OutputPtr>
inline void copy_n(gt::space::cuda tag_in, gt::space::cuda tag_out, InputPtr in,
                   size_type count, OutputPtr out)
{
  gtGpuCheck(cudaMemcpy(
    gt::raw_pointer_cast(out), gt::raw_pointer_cast(in),
    sizeof(typename gt::pointer_traits<InputPtr>::element_type) * count,
    cudaMemcpyDeviceToDevice));
}

template <typename InputPtr, typename OutputPtr>
inline void copy_n(gt::space::cuda tag_in, gt::space::host tag_out, InputPtr in,
                   size_type count, OutputPtr out)
{
  gtGpuCheck(cudaMemcpy(
    gt::raw_pointer_cast(out), gt::raw_pointer_cast(in),
    sizeof(typename gt::pointer_traits<InputPtr>::element_type) * count,
    cudaMemcpyDeviceToHost));
}

template <typename InputPtr, typename OutputPtr>
inline void copy_n(gt::space::host tag_in, gt::space::cuda tag_out, InputPtr in,
                   size_type count, OutputPtr out)
{
  gtGpuCheck(cudaMemcpy(
    gt::raw_pointer_cast(out), gt::raw_pointer_cast(in),
    sizeof(typename gt::pointer_traits<InputPtr>::element_type) * count,
    cudaMemcpyHostToDevice));
}

#if 0 // handled generically instead for host->host copies
template <typename InputPtr, typename OutputPtr>
inline void copy_n(gt::space::host tag_in, gt::space::host tag_out, InputPtr in,
                   size_type count, OutputPtr out)
{
  gtGpuCheck(cudaMemcpy(
    gt::raw_pointer_cast(out), gt::raw_pointer_cast(in),
    sizeof(typename gt::pointer_traits<InputPtr>::element_type) * count,
    cudaMemcpyHostToHost));
}
#endif

} // namespace copy_impl

namespace fill_impl
{
template <typename Ptr, typename T>
inline void fill(gt::space::cuda tag, Ptr first, Ptr last, const T& value)
{
  using element_type = typename gt::pointer_traits<Ptr>::element_type;
  if (element_type(value) == element_type()) {
    gtGpuCheck(cudaMemset(gt::raw_pointer_cast(first), 0,
                          (last - first) * sizeof(element_type)));
  } else {
    assert(sizeof(element_type) == 1);
    gtGpuCheck(cudaMemset(gt::raw_pointer_cast(first), value,
                          (last - first) * sizeof(element_type)));
  }
}
} // namespace fill_impl

template <>
class backend_ops<gt::space::cuda>
{
public:
  static void device_synchronize() { gtGpuCheck(cudaStreamSynchronize(0)); }

  static int device_get_count()
  {
    int device_count;
    gtGpuCheck(cudaGetDeviceCount(&device_count));
    return device_count;
  }

  static void device_set(int device_id)
  {
    gtGpuCheck(cudaSetDevice(device_id));
  }

  static int device_get()
  {
    int device_id;
    gtGpuCheck(cudaGetDevice(&device_id));
    return device_id;
  }

  static uint32_t device_get_vendor_id(int device_id)
  {
    cudaDeviceProp prop;
    uint32_t packed = 0;

    gtGpuCheck(cudaGetDeviceProperties(&prop, device_id));

    packed |= (0x000000FF & ((uint32_t)prop.pciDeviceID));
    packed |= (0x0000FF00 & (((uint32_t)prop.pciBusID) << 8));
    packed |= (0xFFFF0000 & (((uint32_t)prop.pciDomainID) << 16));

    return packed;
  }

  template <typename Ptr>
  static bool is_device_accessible(const Ptr p)
  {
    cudaPointerAttributes attr;
    cudaError_t rval = cudaPointerGetAttributes(&attr, p);
    if (rval == cudaErrorInvalidValue) {
      return false;
    }
    gtGpuCheck(rval);
    return (attr.type == cudaMemoryTypeDevice ||
            attr.type == cudaMemoryTypeManaged);
  }

  template <typename Ptr>
  static memory_type get_memory_type(Ptr ptr)
  {
    cudaPointerAttributes attr;
    auto rc = cudaPointerGetAttributes(&attr, ptr);
    if (rc == cudaErrorInvalidValue) {
      cudaGetLastError(); // clear the error
      return memory_type::unregistered;
    }
    gtGpuCheck(rc);
    switch (attr.type) {
      case cudaMemoryTypeHost: return memory_type::host;
      case cudaMemoryTypeDevice: return memory_type::device;
      case cudaMemoryTypeManaged: return memory_type::managed;
      default:
        fprintf(stderr, "ERROR: unknown memoryType %d.\n", attr.type);
        std::abort();
    }
    return static_cast<memory_type>(attr.type);
  }

  template <typename T>
  static void prefetch_device(T* p, size_type n)
  {
    int device_id;
    gtGpuCheck(cudaGetDevice(&device_id));
    gtGpuCheck(cudaMemPrefetchAsync(p, n * sizeof(T), device_id, nullptr));
  }

  template <typename T>
  static void prefetch_host(T* p, size_type n)
  {
    gtGpuCheck(
      cudaMemPrefetchAsync(p, n * sizeof(T), cudaCpuDeviceId, nullptr));
  }

  template <typename T>
  static void copy_async_dd(const T* src, T* dst, size_type count)
  {
    gtGpuCheck(
      cudaMemcpyAsync(dst, src, sizeof(T) * count, cudaMemcpyDeviceToDevice));
  }

  class stream_view : public stream_interface::stream_view_base<cudaStream_t>
  {
  public:
    using base_type = stream_view_base<cudaStream_t>;
    using base_type::base_type;

    stream_view() : base_type(cudaStreamDefault) {}

    bool is_default() { return this->stream_ == cudaStreamDefault; };

    void synchronize() { gtGpuCheck(cudaStreamSynchronize(this->stream_)); }

    auto get_execution_policy() { return thrust::cuda::par.on(this->stream_); }
  };
};

namespace stream_interface
{

template <>
inline cudaStream_t create<cudaStream_t>()
{
  cudaStream_t s;
  gtGpuCheck(cudaStreamCreate(&s));
  return s;
}

template <>
inline void destroy<cudaStream_t>(cudaStream_t s)
{
  gtGpuCheck(cudaStreamDestroy(s));
}

} // namespace stream_interface

} // namespace backend

using stream = backend::stream_interface::stream_base<
  cudaStream_t, backend::backend_ops<gt::space::cuda>::stream_view>;

} // namespace gt

#endif // GTENSOR_BACKEND_CUDA_H
