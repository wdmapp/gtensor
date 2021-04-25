#include <gtest/gtest.h>

#include <stdint.h>

#include <gtensor/gtensor.h>

#include "test_debug.h"

#define MAX_DEVICES 100

#ifdef GTENSOR_HAVE_DEVICE

// NOTE: device management API doesn't have SYCL impl yet
#ifndef GTENSOR_DEVICE_SYCL

TEST(device_backend, list_devices)
{
  int n_devices = gt::backend::device_get_count();
  uint32_t vendor_id[MAX_DEVICES];

  ASSERT_LE(n_devices, MAX_DEVICES);

  for (int i = 0; i < n_devices; i++) {
    vendor_id[i] = gt::backend::device_get_vendor_id(i);
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
TEST(device_backend, managed_allocate)
{
  double* a = gt::backend::standard::gallocator::managed::allocate<double>(N);
  for (int i = 0; i < N; i++) {
    a[i] = ((double)i) / N;
  }
  auto aview = gt::adapt_device(a, gt::shape(N));
  aview = aview + 1.0;
  gt::synchronize();
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(a[i], 1.0 + ((double)i) / N);
  }
  gt::backend::standard::gallocator::managed::deallocate(a);
}

#endif
