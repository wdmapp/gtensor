
#ifndef GTENSOR_BACKEND_CUDA_H
#define GTENSOR_BACKEND_CUDA_H

#include "backend_common.h"

#include <cuda_runtime_api.h>

//#include "thrust/cuda/system/execution_policy.h"
#include "thrust/execution_policy.h"

// ======================================================================
// gt::backend::cuda

namespace gt
{
namespace backend
{

namespace allocator_impl
{

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
  static T* allocate(size_t n)
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

} // namespace allocator_impl

namespace cuda
{

inline void device_synchronize()
{
  gtGpuCheck(cudaStreamSynchronize(0));
}

template <typename T>
inline void device_copy_async_dd(const T* src, T* dst, size_type count)
{
  gtGpuCheck(
    cudaMemcpyAsync(dst, src, sizeof(T) * count, cudaMemcpyDeviceToDevice));
}

inline int device_get_count()
{
  int device_count;
  gtGpuCheck(cudaGetDeviceCount(&device_count));
  return device_count;
}

inline void device_set(int device_id)
{
  gtGpuCheck(cudaSetDevice(device_id));
}

inline int device_get()
{
  int device_id;
  gtGpuCheck(cudaGetDevice(&device_id));
  return device_id;
}

inline uint32_t device_get_vendor_id(int device_id)
{
  cudaDeviceProp prop;
  uint32_t packed = 0;

  gtGpuCheck(cudaGetDeviceProperties(&prop, device_id));

  packed |= (0x000000FF & ((uint32_t)prop.pciDeviceID));
  packed |= (0x0000FF00 & (((uint32_t)prop.pciBusID) << 8));
  packed |= (0xFFFF0000 & (((uint32_t)prop.pciDomainID) << 16));

  return packed;
}

} // namespace cuda

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
  template <typename Ptr>
  static bool is_device_address(const Ptr p)
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
      default: assert(0);
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
inline cudaStream_t get_default<cudaStream_t>()
{
  return nullptr;
}

template <>
inline void destroy<cudaStream_t>(cudaStream_t s)
{
  gtGpuCheck(cudaStreamDestroy(s));
}

template <>
inline bool is_default<cudaStream_t>(cudaStream_t s)
{
  return s == nullptr;
}

template <>
inline void synchronize<cudaStream_t>(cudaStream_t s)
{
  gtGpuCheck(cudaStreamSynchronize(s));
}

template <typename Stream>
class stream_view_cuda : public stream_view_base<Stream>
{
public:
  using base_type = stream_view_base<Stream>;
  using base_type::base_type;
  using base_type::stream_;

  auto get_execution_policy() { return thrust::cuda::par.on(stream_); }
};

} // namespace stream_interface

} // namespace backend

using stream_view = backend::stream_interface::stream_view_cuda<cudaStream_t>;
using stream =
  backend::stream_interface::stream_base<cudaStream_t, stream_view>;

} // namespace gt

#endif // GTENSOR_BACKEND_CUDA_H
