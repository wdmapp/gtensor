#ifndef GTENSOR_DEVICE_BACKEND_H
#define GTENSOR_DEVICE_BACKEND_H

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <type_traits>

#ifdef GTENSOR_HAVE_DEVICE
#include "device_runtime.h"

#ifdef GTENSOR_USE_THRUST
#include "thrust_ext.h"
#endif

#include "macros.h"

#endif // GTENSOR_HAVE_DEVICE

namespace gt {

namespace backend {

#ifdef GTENSOR_DEVICE_CUDA

void device_memcpy_hh(void *dst, const void *src, size_t bytes)
{
  gtGpuCheck(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToHost));
  gtGpuCheck(cudaDeviceSynchronize());
}

void device_memcpy_dd(void *dst, const void *src, size_t bytes)
{
  gtGpuCheck(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
  gtGpuCheck(cudaDeviceSynchronize());
}

void device_memcpy_dh(void *dst, const void *src, size_t bytes)
{
  gtGpuCheck(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
  gtGpuCheck(cudaDeviceSynchronize());
}

void device_memcpy_hd(void *dst, const void *src, size_t bytes)
{
  gtGpuCheck(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
  gtGpuCheck(cudaDeviceSynchronize());
}

template<typename T>
struct device_allocator
{
  static T* allocate(int count)
  {
    T *p;
    gtGpuCheck(cudaMalloc(&p, sizeof(T) * count));
    return p;
  }

  static void deallocate(T* p)
  {
    if (p != nullptr) {
      gtGpuCheck(cudaFree(p));
    }
  }

  static void memcpy(void *dst, const void *src, std::size_t bytes)
  {
    device_memcpy_dd(dst, src, bytes);
  }
};

template<typename T>
struct host_allocator
{
  static T* allocate(int count)
  {
    T *p;
    gtGpuCheck(cudaMallocHost(&p, sizeof(T) * count));
    return p;
  }

  static void deallocate(T* p)
  {
    if (p != nullptr) {
      gtGpuCheck(cudaFreeHost(p));
    }
  }

  static void memcpy(void *dst, const void *src, std::size_t bytes)
  {
    device_memcpy_hh(dst, src, bytes);
  }
};

#elif defined(GTENSOR_DEVICE_HIP)

void device_memcpy_hh(void *dst, const void *src, size_t bytes)
{
  gtGpuCheck(hipMemcpy(dst, src, bytes, hipMemcpyHostToHost));
  gtGpuCheck(hipDeviceSynchronize());
}

void device_memcpy_dd(void *dst, const void *src, size_t bytes)
{
  gtGpuCheck(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice));
  gtGpuCheck(hipDeviceSynchronize());
}

void device_memcpy_dh(void *dst, const void *src, size_t bytes)
{
  gtGpuCheck(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost));
  gtGpuCheck(hipDeviceSynchronize());
}

void device_memcpy_hd(void *dst, const void *src, size_t bytes)
{
  gtGpuCheck(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice));
  gtGpuCheck(hipDeviceSynchronize());
}

template<typename T>
struct device_allocator
{
  static T* allocate(int count)
  {
    T *p;
    gtGpuCheck(hipMalloc(&p, sizeof(T) * count));
    return p;
  }

  static void deallocate(T* p)
  {
    if (p != nullptr) {
      gtGpuCheck(hipFree(p));
    }
  }

  static void memcpy(void *dst, const void *src, std::size_t bytes)
  {
    device_memcpy_dd(dst, src, bytes);
  }
};

template<typename T>
struct host_allocator
{
  static T* allocate(int count)
  {
    T *p;
    gtGpuCheck(hipHostMalloc(&p, sizeof(T) * count, hipHostMallocDefault));
    return p;
  }

  static void deallocate(T* p)
  {
    if (p != nullptr) {
      gtGpuCheck(hipHostFree(p));
    }
  }

  static void memcpy(void *dst, const void *src, std::size_t bytes)
  {
    device_memcpy_hh(dst, src, bytes);
  }
};

#endif // GTENSOR_DEVICE_HIP

#ifdef GTENSOR_DEVICE_HOST

template<typename T>
struct host_allocator
{
  static T* allocate(int count)
  {
    T *p = static_cast<T*>(malloc(sizeof(T) * count));
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

  static void memcpy(void *dst, const void *src, std::size_t bytes)
  {
    std::memcpy(dst, src, bytes);
  }
};

#endif

#ifdef GTENSOR_USE_THRUST

template<typename Pointer>
inline auto raw_pointer_cast(Pointer p) {
  return thrust::raw_pointer_cast(p);
}

template<typename Pointer>
inline auto device_pointer_cast(Pointer p) {
  return thrust::device_pointer_cast(p);
}

#else // using gt::backend::device_storage

// define no-op device_pointer/raw ponter casts
template<typename Pointer>
inline Pointer raw_pointer_cast(Pointer p) {
  return p;
}

template<typename Pointer>
inline Pointer device_pointer_cast(Pointer p) {
  return p;
}

#endif // GTENSOR_USE_THRUST

} // end namespace backend

} // end namespace gt

#endif // GENSOR_DEVICE_BACKEND_H
