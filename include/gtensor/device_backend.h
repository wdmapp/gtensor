#ifndef GTENSOR_DEVICE_BACKEND_H
#define GTENSOR_DEVICE_BACKEND_H

#include <cstdlib>
#include <iostream>

#ifdef GTENSOR_HAVE_DEVICE
#include "device_runtime.h"

#ifdef GTENSOR_USE_THRUST
#include "thrust_ext.h"
#endif

namespace gt {

namespace backend {

#ifdef GTENSOR_DEVICE_CUDA

inline void checkCall(cudaError_t err, const char *info)
{
  if (err != cudaSuccess) {
    std::cerr << "CUDA error [" << info << "]: "
              << cudaGetErrorString(err) << std::endl;
    std::abort();
  }
}

template<typename T>
struct device_allocator
{
  static T* allocate(int count)
  {
    T *p;
    checkCall(cudaMalloc(&p, sizeof(T) * count), "device_malloc");
    return p;
  }

  static void deallocate(T* p)
  {
    checkCall(cudaFree(p), "device_free");
  }
};

template<typename T>
struct host_allocator
{
  static T* allocate(int count)
  {
    T *p;
    checkCall(cudaMallocHost(&p, sizeof(T) * count), "host_malloc");
    return p;
  }

  static void deallocate(T* p)
  {
    checkCall(cudaFreeHost(p), "host_free");
  }
};

void device_memcpy_dd(void *dst, const void *src, size_t bytes)
{
  checkCall(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice),
            "device_memcpy_dd memcpy");
  checkCall(cudaDeviceSynchronize(), "device_memcpy_dd synchronize");
}

void device_memcpy_dh(void *dst, const void *src, size_t bytes)
{
  checkCall(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost),
            "device_memcpy_dh memcpy");
  checkCall(cudaDeviceSynchronize(), "device_memcpy_dh synchronize");
}

void device_memcpy_hd(void *dst, const void *src, size_t bytes)
{
  checkCall(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice),
            "device_memcpy_hd memcpy");
  checkCall(cudaDeviceSynchronize(), "device_memcpy_hd synchronize");
}

#elif defined(GTENSOR_DEVICE_HIP)

inline void checkCall(hipError_t err, const char *info)
{
  if (err != hipSuccess) {
    std::cerr << "HIP error [" << info << "]: "
              << hipGetErrorString(err) << std::endl;
    std::abort();
  }
}

template<typename T>
struct device_allocator
{
  static T* allocate(int count)
  {
    T *p;
    checkCall(hipMalloc(&p, sizeof(T) * count), "device_malloc");
    return p;
  }

  static void deallocate(T* p)
  {
    checkCall(hipFree(p), "device_free");
  }
};

template<typename T>
struct host_allocator
{
  static T* allocate(int count)
  {
    T *p;
    checkCall(hipHostMalloc(&p, sizeof(T) * count, hipHostMallocDefault),
              "host_malloc");
    return p;
  }

  static void deallocate(T* p)
  {
    checkCall(hipHostFree(p), "host_free");
  }
};

void device_memcpy_dd(void *dst, const void *src, size_t bytes)
{
  checkCall(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice),
            "device_memcpy_dd memcpy");
  checkCall(hipDeviceSynchronize(), "device_memcpy_dd synchronize");
}

void device_memcpy_dh(void *dst, const void *src, size_t bytes)
{
  checkCall(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost),
            "device_memcpy_dh memcpy");
  checkCall(hipDeviceSynchronize(), "device_memcpy_dh synchronize");
}

void device_memcpy_hd(void *dst, const void *src, size_t bytes)
{
  checkCall(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice),
            "device_memcpy_hd memcpy");
  checkCall(hipDeviceSynchronize(), "device_memcpy_hd synchronize");
}

#endif // GTENSOR_DEVICE_HIP

#ifdef GTENSOR_USE_THRUST

// define no-op device_pointer/raw ponter casts
template<typename Pointer>
inline Pointer raw_pointer_cast(Pointer p) {
  return thrust::raw_pointer_cast(p);
}

template<typename Pointer>
inline Pointer device_pointer_cast(Pointer p) {
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

#endif


} // end namespace backend

} // end namespace gt

#endif // GTENSOR_HAVE_DEVICE

#endif // GENSOR_DEVICE_BACKEND_H
