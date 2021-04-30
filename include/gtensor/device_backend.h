#ifndef GTENSOR_DEVICE_BACKEND_H
#define GTENSOR_DEVICE_BACKEND_H

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <type_traits>

#ifdef GTENSOR_HAVE_DEVICE
#include "device_runtime.h"

#ifdef GTENSOR_USE_THRUST
#include <thrust/device_allocator.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#endif

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

namespace space
{

struct host;
#ifdef GTENSOR_USE_THRUST
struct thrust;
#endif
#ifdef GTENSOR_DEVICE_CUDA
struct cuda;
#endif
#ifdef GTENSOR_DEVICE_HIP
struct hip;
#endif
#ifdef GTENSOR_DEVICE_SYCL
struct sycl;
#endif

#ifdef GTENSOR_HAVE_DEVICE

#if GTENSOR_USE_THRUST
using device = thrust;
#elif GTENSOR_DEVICE_CUDA
using device = cuda;
#elif GTENSOR_DEVICE_HIP
using device = hip;
#elif GTENSOR_DEVICE_SYCL
using device = sycl;
#endif

#else // !  GTENSOR_HAVE_DEVICE

using device = host;

#endif

template <typename S>
struct space_traits;

} // namespace space

namespace backend
{

// ======================================================================

template <typename T, typename A, typename S>
struct wrap_allocator
{
  using value_type = T;
  using pointer = typename gt::space::space_traits<S>::template pointer<T>;
  using size_type = gt::size_type;

  pointer allocate(size_type n) { return A::template allocate<T>(n); }
  void deallocate(pointer p, size_type n) { A::deallocate(p); }
};

// ======================================================================
// backend::cuda

#ifdef GTENSOR_DEVICE_CUDA

namespace cuda
{

namespace detail
{

template <typename S_src, typename S_to>
struct copy;

template <>
struct copy<space::device, space::device>
{
  template <typename T>
  static void run(const T* src, T* dst, size_type count)
  {
    gtGpuCheck(
      cudaMemcpy(dst, src, sizeof(T) * count, cudaMemcpyDeviceToDevice));
  }
};

template <>
struct copy<space::device, space::host>
{
  template <typename T>
  static void run(const T* src, T* dst, size_type count)
  {
    gtGpuCheck(cudaMemcpy(dst, src, sizeof(T) * count, cudaMemcpyDeviceToHost));
  }
};

template <>
struct copy<space::host, space::device>
{
  template <typename T>
  static void run(const T* src, T* dst, size_type count)
  {
    gtGpuCheck(cudaMemcpy(dst, src, sizeof(T) * count, cudaMemcpyHostToDevice));
  }
};

template <>
struct copy<space::host, space::host>
{
  template <typename T>
  static void run(const T* src, T* dst, size_type count)
  {
    gtGpuCheck(cudaMemcpy(dst, src, sizeof(T) * count, cudaMemcpyHostToHost));
  }
};

} // namespace detail

template <typename S_src, typename S_to, typename T>
inline void copy(const T* src, T* dst, gt::size_type count)
{
  return detail::copy<S_src, S_to>::run(src, dst, count);
}

struct ops
{
  static void memset(void* dst, int value, gt::size_type nbytes)
  {
    gtGpuCheck(cudaMemset(dst, value, nbytes));
  }
};

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

template <typename T>
using device_allocator =
  wrap_allocator<T, typename gallocator::device, gt::space::device>;

template <typename T>
using host_allocator =
  wrap_allocator<T, typename gallocator::host, gt::space::host>;

template <typename Pointer>
GT_INLINE Pointer raw_pointer_cast(Pointer p)
{
  return p;
}

template <typename Pointer>
GT_INLINE Pointer device_pointer_cast(Pointer p)
{
  return p;
}

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

#endif

// ======================================================================
// backend::hip

#ifdef GTENSOR_DEVICE_HIP

namespace hip
{

namespace detail
{

template <typename S_src, typename S_to>
struct copy;

template <>
struct copy<space::device, space::device>
{
  template <typename T>
  static void run(const T* src, T* dst, size_type count)
  {
    gtGpuCheck(hipMemcpy(dst, src, sizeof(T) * count, hipMemcpyDeviceToDevice));
  }
};

template <>
struct copy<space::device, space::host>
{
  template <typename T>
  static void run(const T* src, T* dst, size_type count)
  {
    gtGpuCheck(hipMemcpy(dst, src, sizeof(T) * count, hipMemcpyHostToDevice));
  }
};

template <>
struct copy<space::host, space::device>
{
  template <typename T>
  static void run(const T* src, T* dst, size_type count)
  {
    gtGpuCheck(hipMemcpy(dst, src, sizeof(T) * count, hipMemcpyHostToDevice));
  }
};

template <>
struct copy<space::host, space::host>
{
  template <typename T>
  static void run(const T* src, T* dst, size_type count)
  {
    gtGpuCheck(hipMemcpy(dst, src, sizeof(T) * count, hipMemcpyHostToHost));
  }
};

} // namespace detail

template <typename S_src, typename S_to, typename T>
inline void copy(const T* src, T* dst, gt::size_type count)
{
  return detail::copy<S_src, S_to>::run(src, dst, count);
}

struct ops
{
  static void memset(void* dst, int value, gt::size_type nbytes)
  {
    gtGpuCheck(hipMemset(dst, value, nbytes));
  }
};

namespace gallocator
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

} // namespace gallocator

template <typename T>
using device_allocator =
  wrap_allocator<T, typename gallocator::device, gt::space::device>;

template <typename T>
using host_allocator =
  wrap_allocator<T, typename gallocator::host, gt::space::host>;

template <typename Pointer>
GT_INLINE Pointer raw_pointer_cast(Pointer p)
{
  return p;
}

template <typename Pointer>
GT_INLINE Pointer device_pointer_cast(Pointer p)
{
  return p;
}

inline void device_synchronize()
{
  gtGpuCheck(hipStreamSynchronize(0));
}

template <typename T>
inline void device_copy_async_dd(const T* src, T* dst, gt::size_type count)
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

#endif

// ======================================================================
// backend::sycl

#ifdef GTENSOR_DEVICE_SYCL

namespace sycl
{

template <typename S_src, typename S_to, typename T>
inline void copy(const T* src, T* dst, gt::size_type count)
{
  cl::sycl::queue& q = gt::backend::sycl::get_queue();
  q.memcpy(dst, src, sizeof(T) * count);
  q.wait();
}

struct ops
{
  static void memset(void* dst, int value, gt::size_type nbytes)
  {
    cl::sycl::queue& q = gt::backend::sycl::get_queue();
    q.memset(dst, value, nbytes);
  }
};

namespace gallocator
{
struct device
{
  template <typename T>
  static T* allocate(size_type n)
  {
    return cl::sycl::malloc_shared<T>(n, gt::backend::sycl::get_queue());
  }

  template <typename T>
  static void deallocate(T* p)
  {
    cl::sycl::free(p, gt::backend::sycl::get_queue());
  }
};

struct managed
{
  template <typename T>
  static T* allocate(size_t n)
  {
    return cl::sycl::malloc_shared<T>(n, gt::backend::sycl::get_queue());
  }

  template <typename T>
  static void deallocate(T* p)
  {
    cl::sycl::free(p, gt::backend::sycl::get_queue());
  }
};

// The host allocation type in SYCL allows device code to directly access
// the data. This is generally not necessary or effecient for gtensor, so
// we opt for the same implementation as for the HOST device below.

struct host
{
  template <typename T>
  static T* allocate(size_type n)
  {
    T* p = static_cast<T*>(malloc(sizeof(T) * n));
    if (!p) {
      std::cerr << "host allocate failed" << std::endl;
      std::abort();
    }
    return p;
  }

  template <typename T>
  static void deallocate(T* p)
  {
    free(p);
  }
};

// template <typename T>
// struct host
// {
//   static T* allocate( : size_type count)
//   {
//     return cl::sycl::malloc_host<T>(count, gt::backend::sycl::get_queue());
//   }

//   static void deallocate(T* p)
//   {
//     cl::sycl::free(p, gt::backend::sycl::get_queue());
//   }
// };

} // namespace gallocator

template <typename T>
using device_allocator =
  wrap_allocator<T, typename gallocator::device, gt::space::device>;

template <typename T>
using host_allocator =
  wrap_allocator<T, typename gallocator::host, gt::space::host>;

template <typename Pointer>
GT_INLINE Pointer raw_pointer_cast(Pointer p)
{
  return p;
}

template <typename Pointer>
GT_INLINE Pointer device_pointer_cast(Pointer p)
{
  return p;
}

inline void device_synchronize()
{
  gt::backend::sycl::get_queue().wait();
}

template <typename T>
inline void device_copy_async_dd(const T* src, T* dst, gt::size_type count)
{
  cl::sycl::queue& q = gt::backend::sycl::get_queue();
  q.memcpy(dst, src, sizeof(T) * count);
}

} // namespace sycl

#endif // GTENSOR_DEVICE_SYCL

// ======================================================================
// backend::host

namespace host
{

template <typename T>
using host_allocator = std::allocator<T>;

template <typename S_from, typename S_to, typename T>
inline void copy(const T* src, T* dst, size_type count)
{
  std::copy(src, src + count, dst);
}

template <typename Pointer>
GT_INLINE Pointer raw_pointer_cast(Pointer p)
{
  return p;
}

template <typename Pointer>
GT_INLINE Pointer device_pointer_cast(Pointer p)
{
  return p;
}

inline void device_synchronize()
{
  // no need to synchronize on host
}

}; // namespace host

// ======================================================================
// backend::thrust

#ifdef GTENSOR_USE_THRUST

namespace thrust
{

struct ops
{
  static void memset(void* dst_, int value, size_type nbytes)
  {
    auto dst = ::thrust::device_pointer_cast(static_cast<char*>(dst_));
    ::thrust::fill(dst, dst + nbytes, value);
  }
};

template <typename T>
using host_allocator = std::allocator<T>;

#if GTENSOR_DEVICE_CUDA && THRUST_VERSION <= 100903
template <typename T>
using device_allocator = ::thrust::device_malloc_allocator<T>;
#else
template <typename T>
using device_allocator = ::thrust::device_allocator<T>;
#endif

template <typename S_src, typename S_dst, typename P_src, typename P_dst>
inline void copy(P_src src, P_dst dst, size_type count)
{
  ::thrust::copy(src, src + count, dst);
}

template <typename Pointer>
GT_INLINE auto raw_pointer_cast(Pointer p)
{
  return ::thrust::raw_pointer_cast(p);
}

template <typename Pointer>
GT_INLINE auto device_pointer_cast(Pointer p)
{
  return ::thrust::device_pointer_cast(p);
}

}; // namespace thrust

#endif

// ======================================================================
// system (default) backend

namespace system
{
#ifdef GTENSOR_USE_THRUST
using namespace backend::thrust;
#elif GTENSOR_DEVICE_CUDA
using namespace backend::cuda;
#elif GTENSOR_DEVICE_HIP
using namespace backend::hip;
#elif GTENSOR_DEVICE_SYCL
using namespace backend::sycl;
#elif GTENSOR_DEVICE_HOST
using namespace backend::host;
#endif

} // namespace system

// ======================================================================
// backend being used in clib (ie., the Fortran interface)

namespace clib
{
#if GTENSOR_DEVICE_CUDA
using namespace backend::cuda;
#elif GTENSOR_DEVICE_HIP
using namespace backend::hip;
#elif GTENSOR_DEVICE_SYCL
using namespace backend::sycl;
#else // just for device_synchronize()
using namespace backend::host;
#endif
} // namespace clib

template <typename Pointer>
GT_INLINE auto raw_pointer_cast(Pointer p)
{
  return system::raw_pointer_cast(p);
}

template <typename Pointer>
GT_INLINE auto device_pointer_cast(Pointer p)
{
  return system::device_pointer_cast(p);
}

namespace detail
{
template <typename S>
struct fill;

#ifdef GTENSOR_HAVE_DEVICE
template <>
struct fill<gt::space::device>
{
  template <typename T, typename U>
  static void run(T* first, T* second, const U& value)
  {
    assert(value == T(0) || sizeof(T) == 1);
    system::ops::memset((char*)first, value, (second - first) * sizeof(T));
  }
};
#endif

template <>
struct fill<gt::space::host>
{
  template <typename T, typename U>
  static void run(T* first, T* second, const U& value)
  {
    std::fill(first, second, value);
  }
};
} // namespace detail

template <typename S, typename T, typename U,
          std::enable_if_t<std::is_convertible<U, T>::value, int> = 0>
void fill(T* first, T* second, const U& value)
{
  return detail::fill<S>::run(first, second, value);
}

} // namespace backend

// ======================================================================
// synchronize

void inline synchronize()
{
  gt::backend::clib::device_synchronize();
}

} // namespace gt

#endif // GENSOR_DEVICE_BACKEND_H
