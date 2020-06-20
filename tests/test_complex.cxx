#include <gtest/gtest.h>

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

// compare against device_comlex_multiply test case with nvprof
TEST(complex, device_double_multiply)
{
  using T = double;
  gt::gtensor<T, 2> h_a(gt::shape(3, 2));
  gt::gtensor<T, 2> h_c(h_a.shape());
  gt::gtensor<T, 2> h_r(h_a.shape());

  gt::gtensor_device<T, 2> a(h_a.shape());
  gt::gtensor_device<T, 2> c(h_a.shape());

  // {{11., 12., 13.}, {21., 22., 23.}};
  h_a(0, 0) = T{11.};
  h_a(1, 0) = T{12.};
  h_a(2, 0) = T{13.};
  h_a(0, 1) = T{21.};
  h_a(1, 1) = T{22.};
  h_a(2, 1) = T{23.};

  h_r(0, 0) = T{22.};
  h_r(1, 0) = T{24.};
  h_r(2, 0) = T{26.};
  h_r(0, 1) = T{42.};
  h_r(1, 1) = T{44.};
  h_r(2, 1) = T{46.};

  gt::copy(h_a, a);

  auto Ifn = gt::scalar(2.0);

  auto e = Ifn * a;
  std::cout << "e type: " << typeid(e).name() << " [kernel "
            << typeid(e.to_kernel()).name() << "]\n";
  c = e;
  std::cout << "c type: " << typeid(c).name() << " [kernel "
            << typeid(c.to_kernel()).name() << "]" << std::endl;

  gt::copy(c, h_c);

  EXPECT_EQ(h_c, h_r);
}

// Note: can be run with nvprof / nsys profile to see if thrust kernels
// are called unnecessarily (other than __unititialized_fill which is
// difficult to avoid without ugly hacks).
TEST(complex, device_complex_multiply)
{
  using T = gt::complex<double>;
  auto I = T{0., 1.0};
  gt::gtensor<T, 2> h_a(gt::shape(3, 2));
  gt::gtensor<T, 2> h_r(h_a.shape());
  gt::gtensor<T, 2> h_c(h_a.shape());

  gt::gtensor_device<T, 2> a(h_a.shape());
  gt::gtensor_device<T, 2> c(h_a.shape());

  // {{11., 12., 13.}, {21., 22., 23.}};
  h_a(0, 0) = T{11., 0};
  h_a(1, 0) = T{12., 0};
  h_a(2, 0) = T{13., 0};
  h_a(0, 1) = T{21., 0};
  h_a(1, 1) = T{22., 0};
  h_a(2, 1) = T{23., 0};

  h_r(0, 0) = T{0., 11.};
  h_r(1, 0) = T{0., 12.};
  h_r(2, 0) = T{0., 13.};
  h_r(0, 1) = T{0., 21.};
  h_r(1, 1) = T{0., 22.};
  h_r(2, 1) = T{0., 23.};

  gt::copy(h_a, a);

  auto e = I * a;
  std::cout << "e type: " << typeid(e).name() << " [kernel "
            << typeid(e.to_kernel()).name() << "]\n";
  c = e;
  std::cout << "c type: " << typeid(c).name() << " [kernel "
            << typeid(c.to_kernel()).name() << "]" << std::endl;

  gt::copy(c, h_c);

  EXPECT_EQ(h_c, h_r);
}

// Note: can be run with nvprof / nsys profile to see if thrust kernels
// are called unnecessarily (other than __unititialized_fill which is
// difficult to avoid without ugly hacks).
TEST(complex, device_eval)
{
  using T = gt::complex<double>;
  auto I = T{0., 1.0};
  gt::gtensor<T, 2> h_a(gt::shape(3, 2));
  gt::gtensor<T, 2> h_b(h_a.shape());
  gt::gtensor<T, 2> h_c(h_a.shape());

  gt::gtensor_device<T, 2> a(h_a.shape());
  gt::gtensor_device<T, 2> b(h_b.shape());

  // {{11., 12., 13.}, {21., 22., 23.}};
  h_a(0, 0) = T{11., 0};
  h_a(1, 0) = T{12., 0};
  h_a(2, 0) = T{13., 0};
  h_a(0, 1) = T{21., 0};
  h_a(1, 1) = T{22., 0};
  h_a(2, 1) = T{23., 0};

  h_b(0, 0) = T{-11., 0};
  h_b(1, 0) = T{-12., 0};
  h_b(2, 0) = T{-13., 0};
  h_b(0, 1) = T{-21., 0};
  h_b(1, 1) = T{-22., 0};
  h_b(2, 1) = T{-23., 0};

  gt::copy(h_a, a);
  gt::copy(h_b, b);

  auto e1 = a + I * b;
  std::cout << "e1  type: " << typeid(e1).name() << "\n";
  auto e2 = a + I * a;
  std::cout << "e2  type: " << typeid(e2).name() << "\n";
  auto e = (1. / 2.) * (e1 + e2);
  std::cout << "e   type: " << typeid(e).name() << "\n";
  auto c = eval(e);
  std::cout << "c   type: " << typeid(c).name() << std::endl;

  gt::copy(c, h_c);

  EXPECT_EQ(h_c, h_a);
}

#endif
