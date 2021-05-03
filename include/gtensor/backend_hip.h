
#ifndef GTENSOR_BACKEND_HIP_H
#define GTENSOR_BACKEND_HIP_H

#include "pointer_traits.h"

#include <hip/hip_runtime.h>

// ======================================================================
// gt::backend::hip

namespace gt
{
namespace backend
{

namespace allocator_impl
{
template <>
struct gallocator<gt::space::hip>
{
  struct device
  {
    template <typename T>
    static T* allocate(size_type n)
    {
      T* p;
      gtGpuCheck(hipMalloc(&p, sizeof(T) * n));
      return p;
    }

    template <typename T>
    static void deallocate(T* p)
    {
      gtGpuCheck(hipFree(p));
    }
  };

  struct managed
  {
    template <typename T>
    static T* allocate(size_t n)
    {
      T* p;
      gtGpuCheck(hipMallocManaged(&p, sizeof(T) * n));
      return p;
    }

    template <typename T>
    static void deallocate(T* p)
    {
      gtGpuCheck(hipFree(p));
    }
  };

  struct host
  {
    template <typename T>
    static T* allocate(size_type n)
    {
      T* p;
      gtGpuCheck(hipHostMalloc(&p, sizeof(T) * n));
      return p;
    }

    template <typename T>
    static void deallocate(T* p)
    {
      gtGpuCheck(hipHostFree(p));
    }
  };
};
} // namespace allocator_impl

namespace hip
{

inline void device_synchronize()
{
  gtGpuCheck(hipStreamSynchronize(0));
}

template <typename T>
inline void device_copy_async_dd(const T* src, T* dst, size_type count)
{
  gtGpuCheck(
    hipMemcpyAsync(dst, src, sizeof(T) * count, hipMemcpyDeviceToDevice));
}

inline int device_get_count()
{
  int device_count;
  gtGpuCheck(hipGetDeviceCount(&device_count));
  return device_count;
}

inline void device_set(int device_id)
{
  gtGpuCheck(hipSetDevice(device_id));
}

inline int device_get()
{
  int device_id;
  gtGpuCheck(hipGetDevice(&device_id));
  return device_id;
}

inline uint32_t device_get_vendor_id(int device_id)
{
  hipDeviceProp_t prop;
  uint32_t packed = 0;

  gtGpuCheck(hipGetDeviceProperties(&prop, device_id));

  packed |= (0x000000FF & ((uint32_t)prop.pciDeviceID));
  packed |= (0x0000FF00 & (((uint32_t)prop.pciBusID) << 8));
  packed |= (0xFFFF0000 & (((uint32_t)prop.pciDomainID) << 16));

  return packed;
}

} // namespace hip

namespace allocator_impl
{

template <typename T>
struct selector<T, gt::space::hip>
{
  using type =
    wrap_allocator<T, gallocator<gt::space::hip>::device, gt::space::hip>;
};

#if 0
template <typename T>
struct selector<T, gt::space::host>
{
  using type =
    wrap_allocator<T, gallocator<gt::space::hip>::host, gt::space::host>;
};
#endif

} // namespace allocator_impl

namespace copy_impl
{

template <typename InputPtr, typename OutputPtr>
inline void copy_n(gt::space::hip tag_in, gt::space::hip tag_out, InputPtr in,
                   size_type count, OutputPtr out)
{
  gtGpuCheck(hipMemcpy(
    gt::raw_pointer_cast(out), gt::raw_pointer_cast(in),
    sizeof(typename gt::pointer_traits<InputPtr>::element_type) * count,
    hipMemcpyDeviceToDevice));
}

template <typename InputPtr, typename OutputPtr>
inline void copy_n(gt::space::hip tag_in, gt::space::host tag_out, InputPtr in,
                   size_type count, OutputPtr out)
{
  gtGpuCheck(hipMemcpy(
    gt::raw_pointer_cast(out), gt::raw_pointer_cast(in),
    sizeof(typename gt::pointer_traits<InputPtr>::element_type) * count,
    hipMemcpyDeviceToHost));
}

template <typename InputPtr, typename OutputPtr>
inline void copy_n(gt::space::host tag_in, gt::space::hip tag_out, InputPtr in,
                   size_type count, OutputPtr out)
{
  gtGpuCheck(hipMemcpy(
    gt::raw_pointer_cast(out), gt::raw_pointer_cast(in),
    sizeof(typename gt::pointer_traits<InputPtr>::element_type) * count,
    hipMemcpyHostToDevice));
}

#if 0 // handled generically instead for host->host copies
template <typename InputPtr, typename OutputPtr>
inline void copy_n(gt::space::host tag_in, gt::space::host tag_out, InputPtr in,
                   size_type count, OutputPtr out)
{
  gtGpuCheck(hipMemcpy(
    gt::raw_pointer_cast(out), gt::raw_pointer_cast(in),
    sizeof(typename gt::pointer_traits<InputPtr>::element_type) * count,
    hipMemcpyHostToHost));
}
#endif

} // namespace copy_impl

namespace fill_impl
{
template <typename Ptr, typename T>
inline void fill(gt::space::hip tag, Ptr first, Ptr last, const T& value)
{
  using element_type = typename gt::pointer_traits<Ptr>::element_type;
  if (element_type(value) == element_type()) {
    gtGpuCheck(hipMemset(gt::raw_pointer_cast(first), 0,
                         (last - first) * sizeof(element_type)));
  } else {
    assert(sizeof(element_type) == 1);
    gtGpuCheck(hipMemset(gt::raw_pointer_cast(first), value,
                         (last - first) * sizeof(element_type)));
  }
}
} // namespace fill_impl

} // namespace backend
} // namespace gt

#endif // GTENSOR_BACKEND_HIP_H
