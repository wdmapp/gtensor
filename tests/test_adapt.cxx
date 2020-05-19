#include <gtest/gtest.h>

#include "gtensor/gtensor.h"

#ifdef GTENSOR_HAVE_DEVICE
#include "gtensor/device_runtime.h"

#ifdef __CUDACC__
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
#endif // __CUDACC__

#ifdef __HCC__

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
#endif // __HCC__

#ifdef __SYCL__
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
#endif // __SYCL__

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

