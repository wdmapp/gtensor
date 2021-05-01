#include <memory>

#include <gtest/gtest.h>

#include <thrust/complex.h>
#include <thrust/device_vector.h>

#include "gtensor/thrust_ext.h"

using namespace thrust::ext;

TEST(thrust_ext, is_device_reference)
{
  using D = thrust::device_reference<double>;
  using T = int;

  static_assert(is_device_reference<D>::value, "is_device_reference");
  static_assert(!is_device_reference<T>::value, "is_device_reference");
}

TEST(thrust_ext, has_device_reference)
{
  using D = thrust::device_reference<double>;
  using T = int;
  static_assert(has_device_reference<D, D>::value, "has_device_reference");
  static_assert(has_device_reference<D, T>::value, "has_device_reference");
  static_assert(has_device_reference<T, D>::value, "has_device_reference");
  static_assert(!has_device_reference<T, T>::value, "has_device_reference");
}

TEST(thrust_ext, remove_device_reference_t)
{
  using D = thrust::device_reference<double>;
  using T = int;
  static_assert(std::is_same<remove_device_reference_t<D>, double>::value,
                "remove_device_reference_t");
  static_assert(std::is_same<remove_device_reference_t<T>, int>::value,
                "remove_device_reference_t");
}

TEST(thrust_ext, add_complex_device_reference)
{
  using T = thrust::complex<double>;
  thrust::device_vector<T> v(2);
  v[0] = 1.;
  v[1] = 2.;

  EXPECT_EQ(v[0] + v[1], 3.);
  EXPECT_EQ(v[0] + 3., 4.);
  EXPECT_EQ(3. + v[1], 5.);

  thrust::device_vector<double> d(2);
  d[0] = 1.;
  d[1] = 2.;

  EXPECT_EQ(v[0] + v[1], 3.);
  EXPECT_EQ(v[0] + d[1], 3.);
  EXPECT_EQ(d[0] + v[1], 3.);
}

TEST(thrust_ext, subtract_complex_device_reference)
{
  using T = thrust::complex<double>;
  thrust::device_vector<T> v(2);
  v[0] = 1.;
  v[1] = 2.;

  EXPECT_EQ(v[0] - v[1], -1.);
  EXPECT_EQ(v[0] - 3., -2.);
  EXPECT_EQ(3. - v[1], 1.);
}

TEST(thrust_ext, multiply_complex_device_reference)
{
  using T = thrust::complex<double>;
  thrust::device_vector<T> v(2);
  v[0] = 1.;
  v[1] = 2.;

  EXPECT_EQ(v[0] * v[1], 2.);
  EXPECT_EQ(v[0] * 3., 3.);
  EXPECT_EQ(3. * v[1], 6.);
}

TEST(thrust_ext, divide_complex_device_reference)
{
  using T = thrust::complex<double>;
  thrust::device_vector<T> v(2);
  v[0] = 12.;
  v[1] = 2.;

  EXPECT_EQ(v[0] / v[1], 6.);
  EXPECT_EQ(v[0] / 3., 4.);
  EXPECT_EQ(3. / v[1], 1.5);
}

TEST(thrust_ext, move_construct)
{
  using T = thrust::complex<double>;
  thrust::device_vector<T> v(2);
  v[0] = T(12., 0.);
  v[1] = T(2., 0.);

  auto vp = v.data();
  thrust::device_vector<T> v2(std::move(v));
  auto v2p = v2.data();

  // make sure that no data was copied
  EXPECT_EQ(vp, v2p);
}
