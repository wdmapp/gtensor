#include <gtest/gtest.h>

#include <gtensor/gtensor.h>

#include <gtensor/capi.h>

#include "test_debug.h"

#define MAX_DEVICES 100

#ifdef GTENSOR_HAVE_DEVICE

TEST(clib, list_devices)
{
  int n_devices = gt_backend_device_get_count();
  uint32_t vendor_id[MAX_DEVICES];

  ASSERT_LE(n_devices, MAX_DEVICES);

  for (int i = 0; i < n_devices; i++) {
    vendor_id[i] = gt_backend_device_get_vendor_id(i);
    GT_DEBUG_PRINTLN("device[" << i << "]: 0x" << std::setfill('0')
                               << std::setw(8) << std::hex << vendor_id[i]
                               << std::dec << std::endl);
    for (int j = i - 1; j >= 0; j--) {
      EXPECT_NE(vendor_id[i], vendor_id[j]);
    }
  }
}

#define N 10
TEST(clib, managed_allocate)
{
  double* a = (double*)gt_backend_managed_allocate(N * sizeof(double));
  for (int i = 0; i < N; i++) {
    a[i] = ((double)i) / N;
  }
  auto aview = gt::adapt_device(a, gt::shape(N));
  aview = aview + 1.0;
  gt_synchronize();
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(a[i], 1.0 + ((double)i) / N);
  }
  gt_backend_managed_deallocate((void*)a);
}

#define N 10
TEST(clib, memcpy)
{
  double* h_a = (double*)gt_backend_host_allocate(N * sizeof(double));
  double* d_a = (double*)gt_backend_device_allocate(N * sizeof(double));
  for (int i = 0; i < N; i++) {
    h_a[i] = ((double)i) / N;
  }

  gt_backend_memcpy_hd(d_a, h_a, N * sizeof(double));
  auto aview = gt::adapt_device(d_a, gt::shape(N));
  aview = aview + 1.0;
  gt_synchronize();
  gt_backend_memcpy_dh(h_a, d_a, N * sizeof(double));

  for (int i = 0; i < N; i++) {
    EXPECT_EQ(h_a[i], 1.0 + ((double)i) / N);
  }

  gt_backend_host_deallocate((void*)h_a);
  gt_backend_device_deallocate((void*)d_a);
}

#define N 10
TEST(clib, memcpy_async)
{
  double* a = (double*)gt_backend_managed_allocate(N * sizeof(double));
  double* b = (double*)gt_backend_managed_allocate(N * sizeof(double));
  for (int i = 0; i < N; i++) {
    a[i] = ((double)i) / N;
  }

  gt_backend_memcpy_async_dd(b, a, N * sizeof(double));
  auto bview = gt::adapt_device(b, gt::shape(N));
  bview = bview + 1.0;
  gt_synchronize();

  for (int i = 0; i < N; i++) {
    EXPECT_EQ(b[i], 1.0 + ((double)i) / N);
  }

  gt_backend_managed_deallocate((void*)a);
  gt_backend_managed_deallocate((void*)b);
}

#define N 10
TEST(clib, memset)
{
  uint8_t* d_a = (uint8_t*)gt_backend_device_allocate(N);
  uint8_t* h_a = (uint8_t*)gt_backend_host_allocate(N);
  const int v = 0xDA;

  gt_backend_memset(static_cast<void*>(d_a), v, N);
  gt_synchronize();
  gt_backend_memcpy_dh(h_a, d_a, N);

  for (int i = 0; i < N; i++) {
    EXPECT_EQ(h_a[i], v);
  }

  gt_backend_host_deallocate((void*)h_a);
  gt_backend_device_deallocate((void*)d_a);
}

TEST(clib, is_device_address)
{
  uint8_t* d_a = (uint8_t*)gt_backend_device_allocate(N);
  uint8_t* m_a = (uint8_t*)gt_backend_managed_allocate(N);
  uint8_t* h_a = (uint8_t*)gt_backend_host_allocate(N);
  uint8_t* a = (uint8_t*)malloc(N);

#ifdef GTENSOR_DEVICE_SYCL
  // special case SYCL to handle the host backend, which says that
  // even device pointers are not device addresses (true from a hardware
  // perspective, even if it's logically false in gtensor).
  sycl::device d = gt::backend::sycl::get_queue().get_device();
  if (d.is_gpu() || d.is_cpu()) {
    ASSERT_TRUE(gt_backend_is_device_address(d_a));
    ASSERT_TRUE(gt_backend_is_device_address(d_a + N / 2));
    ASSERT_TRUE(gt_backend_is_device_address(m_a));
    ASSERT_TRUE(gt_backend_is_device_address(m_a + N / 2));
    ASSERT_FALSE(gt_backend_is_device_address(h_a));
    ASSERT_FALSE(gt_backend_is_device_address(h_a + N / 2));
    ASSERT_FALSE(gt_backend_is_device_address(a));
    ASSERT_FALSE(gt_backend_is_device_address(a + N / 2));
  } else {
    // host backend
    ASSERT_FALSE(gt_backend_is_device_address(d_a));
    ASSERT_FALSE(gt_backend_is_device_address(m_a));
    ASSERT_FALSE(gt_backend_is_device_address(h_a));
    ASSERT_FALSE(gt_backend_is_device_address(a));
  }
#else
  ASSERT_TRUE(gt_backend_is_device_address(d_a));
  ASSERT_TRUE(gt_backend_is_device_address(d_a + N / 2));
  ASSERT_TRUE(gt_backend_is_device_address(m_a));
  ASSERT_TRUE(gt_backend_is_device_address(m_a + N / 2));
  ASSERT_FALSE(gt_backend_is_device_address(h_a));
  ASSERT_FALSE(gt_backend_is_device_address(h_a + N / 2));
  ASSERT_FALSE(gt_backend_is_device_address(a));
  ASSERT_FALSE(gt_backend_is_device_address(a + N / 2));
#endif

  gt_backend_host_deallocate((void*)h_a);
  gt_backend_managed_deallocate((void*)m_a);
  gt_backend_device_deallocate((void*)d_a);
  free(a);
}

TEST(clib, mix_managed_host)
{
  double* a = (double*)gt_backend_managed_allocate(N * sizeof(double));
  double* result = (double*)gt_backend_managed_allocate(N * sizeof(double));
  double* h_b = (double*)gt_backend_host_allocate(N * sizeof(double));
  for (int i = 0; i < N; i++) {
    a[i] = ((double)i) / N;
    h_b[i] = 2 * N;
  }

  for (int i = 0; i < N; i++) {
    result[i] = 0.5 * a[i] * h_b[i];
  }

  for (int i = 0; i < N; i++) {
    EXPECT_NEAR(result[i], (double)i, 1e-14);
  }

  gt_backend_managed_deallocate((void*)a);
  gt_backend_managed_deallocate((void*)result);
  gt_backend_host_deallocate((void*)h_b);
}

#endif // GTENSOR_HAVE_DEVICE
