
#ifndef GTENSOR_BACKEND_CUDA_H
#define GTENSOR_BACKEND_CUDA_H

#include "pointer_traits.h"

#include <cuda_runtime_api.h>

// ======================================================================
// gt::backend::cuda

namespace gt
{
namespace backend
{

namespace cuda
{

namespace gallocator
{
using size_type = gt::size_type;

struct device
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

struct managed
{
  template <typename T>
  static T* allocate(size_t n)
  {
    T* p;
    gtGpuCheck(cudaMallocManaged(&p, sizeof(T) * n));
    return p;
  }

  template <typename T>
  static void deallocate(T* p)
  {
    gtGpuCheck(cudaFree(p));
  }
};

struct host
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

} // namespace gallocator

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

namespace allocator_impl
{

template <typename T>
struct selector<T, gt::space::cuda>
{
  using type =
    wrap_allocator<T, typename cuda::gallocator::device, gt::space::cuda>;
};

#if 0
template <typename T>
struct selector<T, gt::space::host>
{
  using type =
    wrap_allocator<T, typename cuda::gallocator::host, gt::space::host>;
};
#endif

} // namespace allocator_impl

namespace copy_impl
{

template <typename InputPtr, typename OutputPtr>
inline void copy_n(gt::space::cuda tag_in, gt::space::cuda tag_out, InputPtr in,
                   size_type count, OutputPtr out)
{
  gtGpuCheck(cudaMemcpy(
    backend::raw_pointer_cast(out), backend::raw_pointer_cast(in),
    sizeof(typename gt::pointer_traits<InputPtr>::element_type) * count,
    cudaMemcpyDeviceToDevice));
}

template <typename InputPtr, typename OutputPtr>
inline void copy_n(gt::space::cuda tag_in, gt::space::host tag_out, InputPtr in,
                   size_type count, OutputPtr out)
{
  gtGpuCheck(cudaMemcpy(
    backend::raw_pointer_cast(out), backend::raw_pointer_cast(in),
    sizeof(typename gt::pointer_traits<InputPtr>::element_type) * count,
    cudaMemcpyDeviceToHost));
}

template <typename InputPtr, typename OutputPtr>
inline void copy_n(gt::space::host tag_in, gt::space::cuda tag_out, InputPtr in,
                   size_type count, OutputPtr out)
{
  gtGpuCheck(cudaMemcpy(
    backend::raw_pointer_cast(out), backend::raw_pointer_cast(in),
    sizeof(typename gt::pointer_traits<InputPtr>::element_type) * count,
    cudaMemcpyHostToDevice));
}

#if 0 // handled generically instead for host->host copies
template <typename InputPtr, typename OutputPtr>
inline void copy_n(gt::space::host tag_in, gt::space::host tag_out, InputPtr in,
                   size_type count, OutputPtr out)
{
  gtGpuCheck(cudaMemcpy(
    backend::raw_pointer_cast(out), backend::raw_pointer_cast(in),
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
  assert(value == T(0) || sizeof(T) == 1);
  gtGpuCheck(cudaMemset(backend::raw_pointer_cast(first), value, last - first));
}
} // namespace fill_impl

} // namespace backend
} // namespace gt

#endif // GTENSOR_BACKEND_CUDA_H