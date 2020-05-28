#include <gtest/gtest.h>

#include "gtensor/gtensor.h"
#include "gtensor/gtensor_storage.h"

TEST(gtensor_storage, host_copy_assign)
{
  constexpr int N = 16;
  gt::backend::host_storage<double> h1(N);
  gt::backend::host_storage<double> h2(N);

  for (int i=0; i<h1.size(); i++) {
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

  for (int i=0; i<h1.size(); i++) {
    h1[i] = (double)i;
  }

  h2 = std::move(h1);

  EXPECT_EQ(h1.size(), 0);
  EXPECT_EQ(h1.data(), nullptr);

  EXPECT_EQ(h2.size(), N);
  for (int i=0; i<N; i++) {
    EXPECT_EQ(h2[i], (double)i);
  }
}

TEST(gtensor_storage, host_move_ctor)
{
  constexpr int N = 16;
  gt::backend::host_storage<double> h1(N);

  for (int i=0; i<h1.size(); i++) {
    h1[i] = (double)i;
  }

  auto h2 = std::move(h1);

  EXPECT_EQ(h1.size(), 0);
  EXPECT_EQ(h1.data(), nullptr);

  EXPECT_EQ(h2.size(), N);
  for (int i=0; i<N; i++) {
    EXPECT_EQ(h2[i], (double)i);
  }
}


#ifdef GTENSOR_HAVE_DEVICE

TEST(gtensor_storage, device_copy_assign)
{
  constexpr int N = 16;
  using T = double;
  gt::backend::host_storage<T> h1(N);
  gt::backend::device_storage<T> d1(N);
  gt::backend::device_storage<T> d2(N);

  for (int i=0; i<h1.size(); i++) {
    h1[i] = static_cast<T>(i);
  }

  gt::backend::device_memcpy_hd(d1.data(), h1.data(), sizeof(T)*h1.size());
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
  for (int i=0; i<h1.size(); i++) {
    h1[i] = static_cast<T>(i);
  }
  gt::backend::device_memcpy_hd(d1.data(), h1.data(), sizeof(T)*h1.size());
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
  for (int i=0; i<h1.size(); i++) {
    h1[i] = static_cast<T>(i);
  }
  gt::backend::device_memcpy_hd(d1.data(), h1.data(), sizeof(T)*h1.size());
  d1_copy = d1;

  auto d2 = std::move(d1);

  EXPECT_EQ(d1_copy.size(), N);

  EXPECT_EQ(d1.size(), 0);
  EXPECT_EQ(d1.data(), nullptr);

  EXPECT_EQ(d2.size(), N);
  EXPECT_EQ(d2, d1_copy);
}

#endif // GTENSOR_HAVE_DEVICE

