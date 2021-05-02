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
#include "pointer_traits.h"
#include "space_forward.h"

namespace gt
{
namespace space
{

// ======================================================================
// space_traits

template <typename S>
struct space_traits;

template <>
struct space_traits<host>
{
  template <typename T>
  using pointer = T*;
};

#ifdef GTENSOR_HAVE_DEVICE

template <>
struct space_traits<device>
{
#ifdef GTENSOR_USE_THRUST
  template <typename T>
  using pointer = ::thrust::device_ptr<T>;
#else
  template <typename T>
  using pointer = gt::device_ptr<T>;
#endif
};

#endif

} // namespace space

namespace backend
{

template <typename P>
GT_INLINE auto raw_pointer_cast(P p)
{
  return gt::pointer_traits<P>::get(p);
}

template <typename T>
GT_INLINE auto device_pointer_cast(T* p)
{
  using pointer =
    typename gt::space::space_traits<gt::space::device>::template pointer<T>;
  return pointer(p);
}

// ======================================================================

template <typename T, typename A, typename S>
struct wrap_allocator
{
  using value_type = T;
  using pointer = typename gt::space::space_traits<S>::template pointer<T>;
  using size_type = gt::size_type;

  pointer allocate(size_type n) { return pointer(A::template allocate<T>(n)); }
  void deallocate(pointer p, size_type n)
  {
    A::deallocate(gt::pointer_traits<pointer>::get(p));
  }
};

// ======================================================================
// backend::cuda

#ifdef GTENSOR_DEVICE_CUDA

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

template <typename T>
using device_allocator =
  wrap_allocator<T, typename gallocator::device, gt::space::device>;

template <typename T>
using host_allocator =
  wrap_allocator<T, typename gallocator::host, gt::space::host>;

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

namespace copy_impl
{
template <typename InputPtr, typename OutputPtr>
inline void copy_n(gt::space::host tag_in, gt::space::host tag_out, InputPtr in,
                   size_type count, OutputPtr out)
{
  std::copy_n(in, count, out);
}
} // namespace copy_impl

namespace fill_impl
{
template <typename Ptr, typename T>
inline void fill(gt::space::host tag, Ptr first, Ptr last, const T& value)
{
  std::fill(first, last, value);
}
} // namespace fill_impl

namespace host
{

template <typename T>
using host_allocator = std::allocator<T>;

template <typename S_from, typename S_to, typename T>
inline void copy(const T* src, T* dst, size_type count)
{
  std::copy(src, src + count, dst);
}

inline void device_synchronize()
{
  // no need to synchronize on host
}

}; // namespace host

// ======================================================================
// backend::thrust

#ifdef GTENSOR_USE_THRUST

namespace copy_impl
{

template <typename InputPtr, typename OutputPtr>
inline void copy_n(gt::space::thrust tag_in, gt::space::thrust tag_out,
                   InputPtr in, size_type count, OutputPtr out)
{
  ::thrust::copy_n(in, count, out);
}

template <typename InputPtr, typename OutputPtr>
inline void copy_n(gt::space::thrust tag_in, gt::space::host tag_out,
                   InputPtr in, size_type count, OutputPtr out)
{
  ::thrust::copy_n(in, count, out);
}

template <typename InputPtr, typename OutputPtr>
inline void copy_n(gt::space::host tag_in, gt::space::thrust tag_out,
                   InputPtr in, size_type count, OutputPtr out)
{
  ::thrust::copy_n(in, count, out);
}

} // namespace copy_impl

namespace fill_impl
{
template <typename Ptr, typename T>
inline void fill(gt::space::thrust tag, Ptr first, Ptr last, const T& value)
{
  ::thrust::fill(first, last, value);
}
} // namespace fill_impl

namespace thrust
{

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

template <
  typename S, typename Ptr, typename T,
  std::enable_if_t<
    std::is_convertible<T, typename pointer_traits<Ptr>::element_type>::value,
    int> = 0>
void fill(Ptr first, Ptr last, const T& value)
{
  return fill_impl::fill(S{}, first, last, value);
}

} // namespace backend

// ======================================================================
// copy_n

template <
  typename InputPtr, typename OutputPtr,
  std::enable_if_t<is_allowed_element_type_conversion<
                     typename pointer_traits<OutputPtr>::element_type,
                     typename pointer_traits<InputPtr>::element_type>::value,
                   int> = 0>
inline void copy_n(InputPtr in, gt::size_type count, OutputPtr out)
{
  return gt::backend::copy_impl::copy_n(
    typename pointer_traits<InputPtr>::space_type{},
    typename pointer_traits<OutputPtr>::space_type{}, in, count, out);
}

// ======================================================================
// synchronize

void inline synchronize()
{
  gt::backend::clib::device_synchronize();
}

} // namespace gt

#endif // GENSOR_DEVICE_BACKEND_H
