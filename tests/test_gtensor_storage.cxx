#include <gtest/gtest.h>

#include "gtensor/gtensor.h"
#include "gtensor/gtensor_storage.h"

#include "test_debug.h"

TEST(gtensor_storage, host_copy_assign)
{
  constexpr int N = 16;
  gt::backend::host_storage<double> h1(N);
  gt::backend::host_storage<double> h2(N);

  for (int i = 0; i < h1.size(); i++) {
    h1[i] = (double)i;
  }
  h2 = h1;

  EXPECT_EQ(h2.size(), N);
  EXPECT_EQ(h2, h1);
}

TEST(gtensor_storage, host_move_assign)
{
  constexpr int N = 16;
  gt::backend::host_storage<double> h1(N);
  gt::backend::host_storage<double> h2;

  for (int i = 0; i < h1.size(); i++) {
    h1[i] = (double)i;
  }

  h2 = std::move(h1);

  EXPECT_EQ(h1.size(), 0);
  EXPECT_EQ(h1.data(), nullptr);

  EXPECT_EQ(h2.size(), N);
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(h2[i], (double)i);
  }
}

TEST(gtensor_storage, host_move_ctor)
{
  constexpr int N = 16;
  gt::backend::host_storage<double> h1(N);

  for (int i = 0; i < h1.size(); i++) {
    h1[i] = (double)i;
  }

  auto h2 = std::move(h1);

  EXPECT_EQ(h1.size(), 0);
  EXPECT_EQ(h1.data(), nullptr);

  EXPECT_EQ(h2.size(), N);
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(h2[i], (double)i);
  }
}

TEST(gtensor_storage, host_resize_from_zero)
{
  constexpr int N = 16;
  gt::backend::host_storage<double> h1{};
  EXPECT_EQ(h1.size(), 0);
  EXPECT_EQ(h1.capacity(), 0);
  EXPECT_EQ(h1.data(), nullptr);

  h1.resize(N);
  EXPECT_EQ(h1.size(), N);
  EXPECT_EQ(h1.capacity(), N);
  EXPECT_NE(h1.data(), nullptr);

  for (int i = 0; i < h1.size(); i++) {
    h1[i] = (double)i;
  }

  EXPECT_EQ(h1.size(), N);
  EXPECT_EQ(h1.capacity(), N);
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(h1[i], (double)i);
  }
}

TEST(gtensor_storage, host_resize_to_zero)
{
  constexpr int N = 16;
  gt::backend::host_storage<double> h1(N);

  for (int i = 0; i < h1.size(); i++) {
    h1[i] = (double)i;
  }

  h1.resize(0);

  EXPECT_EQ(h1.size(), 0);
  EXPECT_EQ(h1.capacity(), N);
  EXPECT_NE(h1.data(), nullptr);
}

TEST(gtensor_storage, host_resize_expand)
{
  constexpr int N = 16;
  constexpr int N2 = 2 * N;
  gt::backend::host_storage<double> h1(N);

  for (int i = 0; i < h1.size(); i++) {
    h1[i] = (double)i;
  }

  h1.resize(N2);

  EXPECT_EQ(h1.size(), N2);
  EXPECT_EQ(h1.capacity(), N2);
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(h1[i], (double)i);
  }
}

TEST(gtensor_storage, host_resize_shrink)
{
  constexpr int N = 16;
  constexpr int N2 = N / 2;
  gt::backend::host_storage<double> h1(N);

  for (int i = 0; i < h1.size(); i++) {
    h1[i] = (double)i;
  }

  h1.resize(N2);

  EXPECT_EQ(h1.size(), N2);
  EXPECT_EQ(h1.capacity(), N);
  for (int i = 0; i < N2; i++) {
    EXPECT_EQ(h1[i], (double)i);
  }
}

TEST(gtensor_storage, type_aliases)
{
  gt::backend::host_storage<double> h1(10);

  GT_DEBUG_TYPE_NAME(decltype(h1)::value_type);
  GT_DEBUG_TYPE_NAME(decltype(h1)::reference);
  GT_DEBUG_TYPE_NAME(decltype(h1)::const_reference);
  GT_DEBUG_TYPE_NAME(decltype(h1)::pointer);
  GT_DEBUG_TYPE_NAME(decltype(h1)::const_pointer);

  EXPECT_TRUE((std::is_same<decltype(h1)::value_type, double>::value));
  EXPECT_TRUE((std::is_same<decltype(h1)::reference, double&>::value));
  EXPECT_TRUE(
    (std::is_same<decltype(h1)::const_reference, const double&>::value));
  EXPECT_TRUE((std::is_same<decltype(h1)::pointer, double*>::value));
  EXPECT_TRUE(
    (std::is_same<decltype(h1)::const_pointer, const double*>::value));
}

#ifdef GTENSOR_HAVE_DEVICE

TEST(gtensor_storage, device_copy_assign)
{
  constexpr int N = 16;
  using T = double;
  gt::backend::host_storage<T> h1(N);
  gt::backend::device_storage<T> d1(N);
  gt::backend::device_storage<T> d2(N);

  for (int i = 0; i < h1.size(); i++) {
    h1[i] = static_cast<T>(i);
  }

  gt::backend::device_copy_hd(h1.data(), d1.data(), h1.size());
  d2 = d1;

  EXPECT_EQ(d2.size(), N);

  EXPECT_EQ(d2, d1);
}

TEST(gtensor_storage, device_move_assign)
{
  constexpr int N = 16;
  using T = double;
  gt::backend::device_storage<T> d1(N);
  gt::backend::device_storage<T> d1_copy(N);
  gt::backend::device_storage<T> d2;

  gt::backend::host_storage<T> h1(N);
  for (int i = 0; i < h1.size(); i++) {
    h1[i] = static_cast<T>(i);
  }
  gt::backend::device_copy_hd(h1.data(), d1.data(), h1.size());
  d1_copy = d1;

  d2 = std::move(d1);

  EXPECT_EQ(d1_copy.size(), N);

  EXPECT_EQ(d1.size(), 0);
  EXPECT_EQ(d1.data(), nullptr);

  EXPECT_EQ(d2.size(), N);
  EXPECT_EQ(d2, d1_copy);
}

TEST(gtensor_storage, device_move_ctor)
{
  constexpr int N = 16;
  using T = double;
  gt::backend::device_storage<T> d1(N);
  gt::backend::device_storage<T> d1_copy(N);

  gt::backend::host_storage<T> h1(N);
  for (int i = 0; i < h1.size(); i++) {
    h1[i] = static_cast<T>(i);
  }
  gt::backend::device_copy_hd(h1.data(), d1.data(), h1.size());
  d1_copy = d1;

  auto d2 = std::move(d1);

  EXPECT_EQ(d1_copy.size(), N);

  EXPECT_EQ(d1.size(), 0);
  EXPECT_EQ(d1.data(), nullptr);

  EXPECT_EQ(d2.size(), N);
  EXPECT_EQ(d2, d1_copy);
}

TEST(gtensor_storage, device_resize_from_zero)
{
  constexpr int N = 16;
  using T = double;
  gt::backend::device_storage<T> d1{};
  gt::backend::host_storage<T> h1(N);
  for (int i = 0; i < h1.size(); i++) {
    h1[i] = static_cast<T>(i);
  }

  EXPECT_EQ(d1.size(), 0);
  EXPECT_EQ(d1.capacity(), 0);
  EXPECT_EQ(d1.data(), nullptr);

  d1.resize(N);
  EXPECT_EQ(d1.size(), N);
  EXPECT_EQ(d1.capacity(), N);
  EXPECT_NE(d1.data(), nullptr);

  // make sure new pointer is viable by copying data to it
  gt::backend::device_copy_hd(h1.data(), d1.data(), h1.size());

  EXPECT_EQ(d1.size(), N);
  EXPECT_EQ(d1.capacity(), N);
}

TEST(gtensor_storage, device_resize_to_zero)
{
  constexpr int N = 16;
  using T = double;
  gt::backend::device_storage<T> d1(N);
  gt::backend::host_storage<T> h1(N);
  for (int i = 0; i < h1.size(); i++) {
    h1[i] = static_cast<T>(i);
  }
  gt::backend::device_copy_hd(h1.data(), d1.data(), h1.size());

  d1.resize(0);

  EXPECT_EQ(d1.size(), 0);
  EXPECT_EQ(d1.capacity(), N);
  EXPECT_NE(d1.data(), nullptr);
}

TEST(gtensor_storage, device_resize_expand)
{
  constexpr int N = 16;
  constexpr int N2 = 2 * N;
  using T = double;
  gt::backend::device_storage<T> d1(N);
  gt::backend::host_storage<T> h1(N);
  for (int i = 0; i < h1.size(); i++) {
    h1[i] = static_cast<T>(i);
  }
  gt::backend::device_copy_hd(h1.data(), d1.data(), h1.size());

  d1.resize(N2);

  EXPECT_EQ(d1.size(), N2);
  EXPECT_EQ(d1.capacity(), N2);

  h1.resize(N2);
  gt::backend::device_copy_hd(d1.data(), h1.data(), N2);
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(h1[i], (double)i);
  }
}

TEST(gtensor_storage, device_resize_shrink)
{
  constexpr int N = 16;
  constexpr int N2 = N / 2;
  using T = double;
  gt::backend::device_storage<T> d1(N);
  gt::backend::host_storage<T> h1(N);
  for (int i = 0; i < h1.size(); i++) {
    h1[i] = static_cast<T>(i);
  }
  gt::backend::device_copy_hd(h1.data(), d1.data(), h1.size());

  d1.resize(N2);

  EXPECT_EQ(d1.size(), N2);
  EXPECT_EQ(d1.capacity(), N);

  gt::backend::device_copy_hd(d1.data(), h1.data(), N2);
  for (int i = 0; i < N2; i++) {
    EXPECT_EQ(h1[i], (double)i);
  }
}

#endif // GTENSOR_HAVE_DEVICE
