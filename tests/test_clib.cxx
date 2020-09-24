#include <gtest/gtest.h>

#include <gtensor/gtensor.h>

#include <gtensor/capi.h>

#include "test_debug.h"

#define MAX_DEVICES 100

#ifdef GTENSOR_HAVE_DEVICE

// NOTE: device management API doesn't have SYCL impl yet
#ifndef GTENSOR_DEVICE_SYCL

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

#endif // not GTENSOR_DEVICE_SYCL

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

#endif // GTENSOR_HAVE_DEVICE
