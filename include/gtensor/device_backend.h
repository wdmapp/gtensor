#ifndef GTENSOR_DEVICE_BACKEND_H
#define GTENSOR_DEVICE_BACKEND_H

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <type_traits>

#ifdef GTENSOR_HAVE_DEVICE
#include "device_runtime.h"

#if defined(GTENSOR_DEVICE_CUDA) || defined(GTENSOR_USE_THRUST)
#include "thrust_ext.h"
#endif

#ifdef GTENSOR_DEVICE_SYCL
#include "sycl_backend.h"
#endif

#endif // GTENSOR_HAVE_DEVICE

#include "defs.h"
#include "macros.h"

namespace gt
{

namespace backend
{

#ifdef GTENSOR_DEVICE_CUDA

inline void device_synchronize()
{
  gtGpuCheck(cudaStreamSynchronize(0));
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

template <typename T>
inline void device_copy_hh(const T* src, T* dst, gt::size_type count)
{
  gtGpuCheck(cudaMemcpy(dst, src, sizeof(T) * count, cudaMemcpyHostToHost));
}

template <typename T>
inline void device_copy_dd(const T* src, T* dst, gt::size_type count)
{
  gtGpuCheck(cudaMemcpy(dst, src, sizeof(T) * count, cudaMemcpyDeviceToDevice));
}

template <typename T>
inline void device_copy_async_dd(const T* src, T* dst, gt::size_type count)
{
  gtGpuCheck(
    cudaMemcpyAsync(dst, src, sizeof(T) * count, cudaMemcpyDeviceToDevice));
}

template <typename T>
inline void device_copy_dh(const T* src, T* dst, gt::size_type count)
{
  gtGpuCheck(cudaMemcpy(dst, src, sizeof(T) * count, cudaMemcpyDeviceToHost));
}

template <typename T>
inline void device_copy_hd(const T* src, T* dst, gt::size_type count)
{
  gtGpuCheck(cudaMemcpy(dst, src, sizeof(T) * count, cudaMemcpyHostToDevice));
}

inline void device_memset(void* dst, int value, gt::size_type nbytes)
{
  gtGpuCheck(cudaMemset(dst, value, nbytes));
}

template <typename T>
struct device_allocator
{
  static T* allocate(size_t count)
  {
    T* p;
    gtGpuCheck(cudaMalloc(&p, sizeof(T) * count));
    return p;
  }

  static void deallocate(T* p)
  {
    if (p != nullptr) {
      gtGpuCheck(cudaFree(p));
    }
  }

  static void copy(const T* src, T* dst, gt::size_type count)
  {
    device_copy_dd(src, dst, count);
  }
};

template <typename T>
struct managed_allocator
{
  static T* allocate(size_t count)
  {
    T* p;
    gtGpuCheck(cudaMallocManaged(&p, sizeof(T) * count));
    return p;
  }

  static void deallocate(T* p)
  {
    if (p != nullptr) {
      gtGpuCheck(cudaFree(p));
    }
  }

  static void copy(const T* src, T* dst, gt::size_type count)
  {
    device_copy_dd(src, dst, count);
  }
};

template <typename T>
struct host_allocator
{
  static T* allocate(gt::size_type count)
  {
    T* p;
    gtGpuCheck(cudaMallocHost(&p, sizeof(T) * count));
    return p;
  }

  static void deallocate(T* p)
  {
    if (p != nullptr) {
      gtGpuCheck(cudaFreeHost(p));
    }
  }

  static void copy(const T* src, T* dst, gt::size_type count)
  {
    device_copy_hh(src, dst, count);
  }
};

#elif defined(GTENSOR_DEVICE_HIP)

inline void device_synchronize()
{
  gtGpuCheck(hipStreamSynchronize(0));
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

template <typename T>
inline void device_copy_hh(const T* src, T* dst, gt::size_type count)
{
  gtGpuCheck(hipMemcpy(dst, src, sizeof(T) * count, hipMemcpyHostToHost));
}

template <typename T>
inline void device_copy_dd(const T* src, T* dst, gt::size_type count)
{
  gtGpuCheck(hipMemcpy(dst, src, sizeof(T) * count, hipMemcpyDeviceToDevice));
}

template <typename T>
inline void device_copy_async_dd(const T* src, T* dst, gt::size_type count)
{
  gtGpuCheck(
    hipMemcpyAsync(dst, src, sizeof(T) * count, hipMemcpyDeviceToDevice));
}

template <typename T>
inline void device_copy_dh(const T* src, T* dst, gt::size_type count)
{
  gtGpuCheck(hipMemcpy(dst, src, sizeof(T) * count, hipMemcpyDeviceToHost));
}

template <typename T>
inline void device_copy_hd(const T* src, T* dst, gt::size_type count)
{
  gtGpuCheck(hipMemcpy(dst, src, sizeof(T) * count, hipMemcpyHostToDevice));
}

inline void device_memset(void* dst, int value, gt::size_type nbytes)
{
  gtGpuCheck(hipMemset(dst, value, nbytes));
}

template <typename T>
struct device_allocator
{
  static T* allocate(gt::size_type count)
  {
    T* p;
    gtGpuCheck(hipMalloc(&p, sizeof(T) * count));
    return p;
  }

  static void deallocate(T* p)
  {
    if (p != nullptr) {
      gtGpuCheck(hipFree(p));
    }
  }

  static void copy(const T* src, T* dst, gt::size_type count)
  {
    device_copy_dd(src, dst, count);
  }
};

template <typename T>
struct managed_allocator
{
  static T* allocate(gt::size_type count)
  {
    T* p;
    gtGpuCheck(hipMallocManaged(&p, sizeof(T) * count));
    return p;
  }

  static void deallocate(T* p)
  {
    if (p != nullptr) {
      gtGpuCheck(hipFree(p));
    }
  }

  static void copy(const T* src, T* dst, gt::size_type count)
  {
    device_copy_dd(src, dst, count);
  }
};

template <typename T>
struct host_allocator
{
  static T* allocate(gt::size_type count)
  {
    T* p;
    gtGpuCheck(hipHostMalloc(&p, sizeof(T) * count, hipHostMallocDefault));
    return p;
  }

  static void deallocate(T* p)
  {
    if (p != nullptr) {
      gtGpuCheck(hipHostFree(p));
    }
  }

  static void copy(const T* src, T* dst, gt::size_type count)
  {
    device_copy_hh(src, dst, count);
  }
};

#elif defined(GTENSOR_DEVICE_SYCL)

inline void device_synchronize()
{
  gt::backend::sycl::get_queue().wait();
}

// TODO: SYCL exception handler
template <typename T>
inline void device_copy(const T* src, T* dst, gt::size_type count)
{
  cl::sycl::queue& q = gt::backend::sycl::get_queue();
  q.memcpy(dst, src, sizeof(T) * count);
  q.wait();
}

template <typename T>
inline void device_copy_hh(const T* src, T* dst, gt::size_type count)
{
  device_copy(src, dst, count);
}

template <typename T>
inline void device_copy_dd(const T* src, T* dst, gt::size_type count)
{
  device_copy(src, dst, count);
}

template <typename T>
inline void device_copy_async_dd(const T* src, T* dst, gt::size_type count)
{
  cl::sycl::queue& q = gt::backend::sycl::get_queue();
  q.memcpy(dst, src, sizeof(T) * count);
}

template <typename T>
inline void device_copy_dh(const T* src, T* dst, gt::size_type count)
{
  device_copy(src, dst, count);
}

template <typename T>
inline void device_copy_hd(const T* src, T* dst, gt::size_type count)
{
  device_copy(src, dst, count);
}

inline void device_memset(void* dst, int value, gt::size_type nbytes)
{
  cl::sycl::queue& q = gt::backend::sycl::get_queue();
  q.memset(dst, value, nbytes);
}

template <typename T>
struct device_allocator
{
  static T* allocate(gt::size_type count)
  {
    return cl::sycl::malloc_device<T>(count, gt::backend::sycl::get_queue());
  }

  static void deallocate(T* p)
  {
    if (p != nullptr) {
      cl::sycl::free(p, gt::backend::sycl::get_queue());
    }
  }

  static void copy(const T* src, T* dst, gt::size_type count)
  {
    device_copy_dd(src, dst, count);
  }
};

template <typename T>
struct managed_allocator
{
  static T* allocate(gt::size_type count)
  {
    return cl::sycl::malloc_shared<T>(count, gt::backend::sycl::get_queue());
  }

  static void deallocate(T* p)
  {
    if (p != nullptr) {
      cl::sycl::free(p, gt::backend::sycl::get_queue());
    }
  }

  static void copy(const T* src, T* dst, gt::size_type count)
  {
    device_copy_dd(src, dst, count);
  }
};

// The host allocation type in SYCL allows device code to directly access
// the data. This is generally not necessary or effecient for gtensor, so
// we opt for the same implementation as for the HOST device below.
template <typename T>
struct host_allocator
{
  static T* allocate(gt::size_type count)
  {
    T* p = static_cast<T*>(malloc(sizeof(T) * count));
    if (p == nullptr) {
      std::cerr << "host allocate failed" << std::endl;
      std::abort();
    }
    return p;
  }

  static void deallocate(T* p)
  {
    if (p != nullptr) {
      free(p);
    }
  }

  static void copy(const T* src, T* dst, gt::size_type count)
  {
    std::memcpy(dst, src, sizeof(T) * count);
  }
};

/*
template<typename T>
struct host_allocator
{
  static T* allocate(gt::size_type count)
  {
    return cl::sycl::malloc_host<T>(count, gt::backend::sycl::get_queue());
  }

  static void deallocate(T* p)
  {
    cl::sycl::free(p, gt::backend::sycl::get_queue());
  }

  static void copy(const void *src, void *dst, gt::size_type count)
  {
    device_copy_hh(dst, src, count);
  }
};
*/

#endif // GTENSOR_DEVICE_{CUDA,HIP,SYCL}

#ifdef GTENSOR_DEVICE_HOST

inline void device_synchronize()
{
  // no need to synchronize on host
}

template <typename T>
struct host_allocator
{
  static T* allocate(gt::size_type count)
  {
    T* p = static_cast<T*>(malloc(sizeof(T) * count));
    if (p == nullptr) {
      std::cerr << "host allocate failed" << std::endl;
      std::abort();
    }
    return p;
  }

  static void deallocate(T* p)
  {
    if (p != nullptr) {
      free(p);
    }
  }

  static void copy(const T* src, T* dst, gt::size_type count)
  {
    std::memcpy(dst, src, count * sizeof(T));
  }
};

#endif

#ifdef GTENSOR_USE_THRUST

template <typename Pointer>
inline auto raw_pointer_cast(Pointer p)
{
  return thrust::raw_pointer_cast(p);
}

template <typename Pointer>
inline auto device_pointer_cast(Pointer p)
{
  return thrust::device_pointer_cast(p);
}

#else // using gt::backend::device_storage

// define no-op device_pointer/raw ponter casts
template <typename Pointer>
inline Pointer raw_pointer_cast(Pointer p)
{
  return p;
}

template <typename Pointer>
inline Pointer device_pointer_cast(Pointer p)
{
  return p;
}

#endif // GTENSOR_USE_THRUST

} // end namespace backend

// ======================================================================
// synchronize

void inline synchronize()
{
  gt::backend::device_synchronize();
}

} // end namespace gt

#endif // GENSOR_DEVICE_BACKEND_H
