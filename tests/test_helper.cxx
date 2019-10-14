
#include <gtest/gtest.h>

#include "gtensor/helper.h"

#include <tuple>

TEST(helper, tuple_max)
{
  std::tuple<> t0;
  std::tuple<int> t1 = {5};
  std::tuple<int, int> t2 = {5, 10};
  std::tuple<int, int, int> t3 = {5, 10, 15};
  std::tuple<int, int, int> t3a = {25, 10, 15};

  auto id = [](auto& val) { return val; };
  EXPECT_EQ(gt::helper::max(id, t0), 0);
  EXPECT_EQ(gt::helper::max(id, t1), 5);
  EXPECT_EQ(gt::helper::max(id, t2), 10);
  EXPECT_EQ(gt::helper::max(id, t3), 15);
  EXPECT_EQ(gt::helper::max(id, t3a), 25);
}
