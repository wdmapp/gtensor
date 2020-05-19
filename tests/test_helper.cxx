
#include <gtest/gtest.h>

#include "gtensor/gtensor.h"
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

TEST(helper, nd_initializer_list)
{
  using namespace gt::helper;

  nd_initializer_list_t<int, 1> nd1 = {1,2,3,4,5,6};
  auto nd1shape = nd_initializer_list_shape<1>(nd1);
  EXPECT_EQ(nd1shape, gt::shape(6));

  nd_initializer_list_t<int, 2> nd2 = {{1,2,3},{4,5,6}};
  auto nd2shape = nd_initializer_list_shape<2>(nd2);
  EXPECT_EQ(nd2shape, gt::shape(3,2));

  nd_initializer_list_t<int, 3> nd3 = {{{1,},{2,},{3,}},
                                       {{4,},{5,},{6,}}};
  auto nd3shape = nd_initializer_list_shape<3>(nd3);
  EXPECT_EQ(nd3shape, gt::shape(1,3,2));
}
