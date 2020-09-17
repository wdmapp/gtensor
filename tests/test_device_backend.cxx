#include <gtest/gtest.h>

#include <stdint.h>

#include <gtensor/gtensor.h>

#include "test_debug.h"

#define MAX_DEVICES 100

#ifdef GTENSOR_HAVE_DEVICE

TEST(device_backend, list_devices)
{
  int n_devices = gt::device_get_count();
  uint32_t vendor_id[MAX_DEVICES];

  ASSERT_LE(n_devices, MAX_DEVICES);

  for (int i = 0; i < n_devices; i++) {
    vendor_id[i] = gt::device_get_vendor_id(i);
    GT_DEBUG_PRINTLN("device[" << i << "]: 0x" << std::setfill('0')
                               << std::setw(8) << std::hex << vendor_id[i]
                               << std::dec << std::endl);
    for (int j = i - 1; j >= 0; j--) {
      EXPECT_NE(vendor_id[i], vendor_id[j]);
    }
  }
}

#endif
