#include <gtest/gtest.h>

#include "gtensor/gtensor.h"

TEST(space, has_space_type)
{
  gt::gtensor<double, 1> a(gt::shape(10));
  auto aspan = a.to_kernel();
  auto aview = a.view(gt::slice(3, 6));

  gt::gtensor_device<double, 1> b(gt::shape(10));
  auto bspan = b.to_kernel();
  auto bview = b.view(gt::slice(3, 6));

  EXPECT_TRUE(gt::has_space_type_host_v<decltype(a)>);
  EXPECT_TRUE(gt::has_space_type_host<decltype(a)>::value);
  EXPECT_TRUE(gt::has_space_type_host_v<decltype(aspan)>);
  EXPECT_TRUE(gt::has_space_type_host<decltype(aspan)>::value);
  EXPECT_TRUE(gt::has_space_type_host_v<decltype(aview)>);
  EXPECT_TRUE(gt::has_space_type_host<decltype(aview)>::value);

  EXPECT_TRUE(gt::has_space_type_device_v<decltype(b)>);
  EXPECT_TRUE(gt::has_space_type_device<decltype(b)>::value);
  EXPECT_TRUE(gt::has_space_type_device_v<decltype(bspan)>);
  EXPECT_TRUE(gt::has_space_type_device<decltype(bspan)>::value);
  EXPECT_TRUE(gt::has_space_type_device_v<decltype(bview)>);
  EXPECT_TRUE(gt::has_space_type_device<decltype(bview)>::value);

#ifdef GTENSOR_HAVE_DEVICE
  EXPECT_FALSE(gt::has_space_type_device_v<decltype(a)>);
  EXPECT_FALSE(gt::has_space_type_device<decltype(a)>::value);
  EXPECT_FALSE(gt::has_space_type_device_v<decltype(aspan)>);
  EXPECT_FALSE(gt::has_space_type_device<decltype(aspan)>::value);
  EXPECT_FALSE(gt::has_space_type_device_v<decltype(aview)>);
  EXPECT_FALSE(gt::has_space_type_device<decltype(aview)>::value);

  EXPECT_FALSE(gt::has_space_type_host_v<decltype(b)>);
  EXPECT_FALSE(gt::has_space_type_host<decltype(b)>::value);
  EXPECT_FALSE(gt::has_space_type_host_v<decltype(bspan)>);
  EXPECT_FALSE(gt::has_space_type_host<decltype(bspan)>::value);
  EXPECT_FALSE(gt::has_space_type_host_v<decltype(bview)>);
  EXPECT_FALSE(gt::has_space_type_host<decltype(bview)>::value);
#else
  // NOTE: if GTENSOR_DEVICE_HOST, gt::space::device == gt::space::host,
  // so the truth values will be true here as well
  EXPECT_TRUE(gt::has_space_type_device_v<decltype(a)>);
  EXPECT_TRUE(gt::has_space_type_device<decltype(a)>::value);
  EXPECT_TRUE(gt::has_space_type_device_v<decltype(aspan)>);
  EXPECT_TRUE(gt::has_space_type_device<decltype(aspan)>::value);
  EXPECT_TRUE(gt::has_space_type_device_v<decltype(aview)>);
  EXPECT_TRUE(gt::has_space_type_device<decltype(aview)>::value);

  EXPECT_TRUE(gt::has_space_type_host_v<decltype(b)>);
  EXPECT_TRUE(gt::has_space_type_host<decltype(b)>::value);
  EXPECT_TRUE(gt::has_space_type_host_v<decltype(bspan)>);
  EXPECT_TRUE(gt::has_space_type_host<decltype(bspan)>::value);
  EXPECT_TRUE(gt::has_space_type_host_v<decltype(bview)>);
  EXPECT_TRUE(gt::has_space_type_host<decltype(bview)>::value);
#endif
}
