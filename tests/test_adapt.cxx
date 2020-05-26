#include <gtest/gtest.h>

#include "gtensor/gtensor.h"

#ifdef GTENSOR_HAVE_DEVICE
#include "gtensor/device_runtime.h"

#ifdef GTENSOR_DEVICE_CUDA
template <typename T>
T* malloc_device(int count)
{
  T *p;
  cudaError_t err = cudaMalloc(&p, sizeof(T) * count);
  return p;
}

template <typename T>
void free_device(T* p)
{
  cudaFree(p);
}
#endif // GTENSOR_DEVICE_CUDA

#ifdef GTENSOR_DEVICE_HIP

template <typename T>
T* malloc_device(int count)
{
  int *p;
  hipError_t err = hipMalloc(&p, sizeof(T) * count);
  return p;
}

template <typename T>
void free_device(T* p)
{
  hipFree(p);
}
#endif // GTENSOR_DEVICE_HIP

#ifdef GTENSOR_DEVICE_SYCL
#include "thrust/sycl.h"

template <typename T>
T* malloc_device(int count)
{
    return sycl::malloc_device<T>(count, thrust::sycl::get_queue());
}

template <typename T>
void free_device(T* p)
{
  sycl::free(p, thrust::sycl::get_queue());
}
#endif // GTENSOR_DEVICE_SYCL

TEST(adapt, adapt_device)
{
  constexpr int N = 10;
  int *a = malloc_device<int>(N);
  auto aview = gt::adapt_device(a, gt::shape(N));

  aview = gt::scalar(7);

  gt::gtensor<int, 1> h_a{gt::shape(N)};

  gt::copy(aview, h_a);

  gt::gtensor<int, 1> h_expected{7,7,7,7,7,7,7,7,7,7};

  EXPECT_EQ(aview, h_expected);

  free_device(a);
}
#endif // GTENSOR_HAVE_DEVICE

