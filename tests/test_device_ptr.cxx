
#include <gtest/gtest.h>

#include <gtensor/device_backend.h>
#include <gtensor/device_ptr.h>
//#include <gtensor/gtensor.h>

#include <array>

#ifdef GTENSOR_HAVE_DEVICE

TEST(device_ptr, ctor)
{
  using T = double;

  gt::device_ptr<T> p0;
  EXPECT_EQ(p0.get(), nullptr);

  T x;
  gt::device_ptr<T> p1(&x);
  EXPECT_EQ(p1.get(), &x);

  gt::device_ptr<T> p2 = p1;
  EXPECT_EQ(p1.get(), &x);
  EXPECT_EQ(p2.get(), &x);
}

TEST(device_ptr, conversion)
{
  using T = double;
  gt::device_ptr<T> p_T;
  gt::device_ptr<const T> p_cT = p_T;
}

TEST(device_ptr, assign)
{
  using T = double;

  T x;
  gt::device_ptr<T> p1(&x);
  EXPECT_EQ(p1.get(), &x);

  gt::device_ptr<T> p2;
  p2 = p1;
  EXPECT_EQ(p2.get(), &x);
}

TEST(device_ptr, get)
{
  using T = double;

  T x;
  gt::device_ptr<T> p1(&x);
  EXPECT_EQ(p1.get(), &x);
}

#if 0
template <typename T>
static void device_ptr_deref(gt::device_ptr<T> p)
{
  // FIXME, 0-d launch would be nicer ;)
  gt::launch<1, gt::space::device>(
    {1}, GT_LAMBDA(int i) { *p = T(99); });
}

TEST(device_ptr, deref)
{
  using T = double;

  gt::backend::system::device_allocator<T> alloc;

  auto p = alloc.allocate(1);
  gt::device_ptr<T> p1(gt::backend::raw_pointer_cast(p));

  device_ptr_deref(p1);

  T h_val;
  gt::copy_n(p, 1, &h_val);
  EXPECT_EQ(h_val, T(99));

  alloc.deallocate(p, 1);
}

template <typename T>
static void device_ptr_access(gt::device_ptr<T> arr, gt::size_type N)
{
  // FIXME, 0-d launch would be nicer ;)
  gt::launch<1, gt::space::device>(
    {N}, GT_LAMBDA(int i) { arr[i] = T(i); });
}

TEST(device_ptr, access)
{
  using T = double;
  const int N = 5;

  gt::backend::system::device_allocator<T> alloc;

  auto p = alloc.allocate(5);
  gt::device_ptr<T> d_arr(gt::backend::raw_pointer_cast(p));

  device_ptr_access(d_arr, N);

  std::array<T, N> h_arr;
  gt::copy_n(p, N, h_arr.data());

  EXPECT_EQ(h_arr, (std::array<T, N>{0, 1, 2, 3, 4}));

  alloc.deallocate(p, N);
}

template <typename T>
static void device_ptr_arrow(gt::device_ptr<T> p)
{
  // FIXME, 0-d launch would be nicer ;)
  gt::launch<1, gt::space::device>(
    {1}, GT_LAMBDA(int i) {
      p->first = 99;
      p->second = 98;
    });
}

TEST(device_ptr, arrow)
{
  using T = std::pair<double, double>;

  gt::backend::system::device_allocator<T> alloc;

  auto p = alloc.allocate(1);
  gt::device_ptr<T> p1(gt::backend::raw_pointer_cast(p));

  device_ptr_arrow(p1);

  T h_val;
  gt::copy(p, 1, &h_val);
  EXPECT_EQ(h_val, T(99, 98));

  alloc.deallocate(p, 1);
}
#endif

#endif // GTENSOR_HAVE_DEVICE
