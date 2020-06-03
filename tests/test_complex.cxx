#include <gtest/gtest.h>

#include <gtensor/complex.h>
#include <gtensor/gtensor.h>

TEST(complex, complex_ops)
{
  using T = gt::complex<double>;
  gt::gtensor<T, 1> h_a(2);
  gt::gtensor<T, 1> h_b(h_a.shape());
  gt::gtensor<T, 1> h_c(h_a.shape());
  gt::gtensor<T, 1> c(h_a.shape());

  h_a(0) = T{7., -2.};
  h_a(1) = T{1., 4.};
  h_b(0) = T{7., 2.};
  h_b(1) = T{1., -4.};

  h_c = h_a + h_b;
  c(0) = T{14., 0.};
  c(1) = T{2., 0.};
  EXPECT_EQ(h_c, c);

  h_c = h_a - h_b;
  c(0) = T{0., -4.};
  c(1) = T{0., 8.};
  EXPECT_EQ(h_c, c);

  h_c = h_a * h_b;
  c(0) = T{53., 0.};
  c(1) = T{17., 0.};
  EXPECT_EQ(h_c, c);
}

#ifdef GTENSOR_HAVE_DEVICE

TEST(complex, device_complex_ops)
{
  using T = gt::complex<double>;
  gt::gtensor<T, 1> h_a(2);
  gt::gtensor<T, 1> h_b(h_a.shape());
  gt::gtensor<T, 1> h_c(h_a.shape());
  gt::gtensor<T, 1> c(h_a.shape());
  gt::gtensor_device<T, 1> d_a(h_a.shape());
  gt::gtensor_device<T, 1> d_b(h_b.shape());
  gt::gtensor_device<T, 1> d_c(h_c.shape());

  h_a(0) = T{7., -2.};
  h_a(1) = T{1., 4.};
  h_b(0) = T{7., 2.};
  h_b(1) = T{1., -4.};

  gt::copy(h_a, d_a);
  gt::copy(h_b, d_b);

  d_c = d_a + d_b;
  gt::copy(d_c, h_c);
  c(0) = T{14., 0.};
  c(1) = T{2., 0.};
  EXPECT_EQ(h_c, c);

  d_c = d_a - d_b;
  gt::copy(d_c, h_c);
  c(0) = T{0., -4.};
  c(1) = T{0., 8.};
  EXPECT_EQ(h_c, c);

  d_c = d_a * d_b;
  gt::copy(d_c, h_c);
  c(0) = T{53., 0.};
  c(1) = T{17., 0.};
  EXPECT_EQ(h_c, c);
}

#endif
