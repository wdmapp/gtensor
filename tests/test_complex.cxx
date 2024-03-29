#include <gtest/gtest.h>

#include <gtensor/gtensor.h>

#include "test_debug.h"

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

TEST(complex, norm)
{
  using T = gt::complex<double>;
  auto one = T{1., 0.};
  auto i = T{0., 1.};

  EXPECT_EQ(gt::norm(one), 1);
  EXPECT_EQ(gt::norm(i), 1);

  gt::gtensor<T, 1> h_a(2);

  h_a(0) = T{1., 1.};
  h_a(1) = T{1., -1.};
  h_a(0) = gt::norm(h_a(0));
  h_a(1) = gt::norm(h_a(1));

  EXPECT_EQ(h_a(0), h_a(1));
}

TEST(complex, abs)
{
  using T = gt::complex<double>;
  auto one = T{1., 0.};
  auto i = T{0., 1.};

  EXPECT_EQ(gt::abs(one), 1);
  EXPECT_EQ(gt::abs(i), 1);

  gt::gtensor<T, 1> h_a(2);

  h_a(0) = T{1., 1.};
  h_a(1) = T{1., -1.};
  h_a(0) = gt::abs(h_a(0));
  h_a(1) = gt::abs(h_a(1));

  EXPECT_EQ(h_a(0), h_a(1));
}

TEST(complex, conj)
{
  using T = gt::complex<double>;
  auto one = T{1., 0.};
  auto i = T{0., 1.};
  auto minus_i = T{0., -1.};
  auto one_one = T{1., 1.};
  auto one_minus_one = T{1., -1.};

  EXPECT_EQ(gt::conj(one), one);
  EXPECT_EQ(gt::conj(i), minus_i);

  gt::gtensor<T, 1> h_a(2);

  h_a(0) = T{1., 1.};
  h_a(1) = T{1., -1.};
  h_a(0) = gt::conj(h_a(0));
  h_a(1) = gt::conj(h_a(1));

  EXPECT_EQ(h_a(0), one_minus_one);
  EXPECT_EQ(h_a(1), one_one);
}

TEST(complex, exp_real)
{
  double x = 1.;

  EXPECT_EQ(gt::exp(x), std::exp(1.));
}

TEST(complex, exp)
{
  using namespace std::complex_literals;
  using T = gt::complex<double>;
  T x = T(0.0, M_PI / 2.0);

  EXPECT_LT(gt::abs(gt::exp(x) - T(0., 1.)), 1e-14);
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

#if defined(GTENSOR_DEVICE_CUDA) || defined(GTENSOR_DEVICE_HIP)

__global__ void kernel_norm(
  gt::gtensor_span_device<gt::complex<double>, 1> d_in,
  gt::gtensor_span_device<double, 1> d_out)
{
  int i = threadIdx.x;
  if (i < d_in.shape(0)) {
    d_out(i) = gt::norm(d_in(i));
  }
}

__global__ void kernel_conj(
  gt::gtensor_span_device<gt::complex<double>, 1> d_in,
  gt::gtensor_span_device<gt::complex<double>, 1> d_out)
{
  int i = threadIdx.x;
  if (i < d_in.shape(0)) {
    d_out(i) = gt::conj(d_in(i));
  }
}

TEST(complex, device_norm)
{
  const int N = 6;
  using T = gt::complex<double>;
  auto I = T{0., 1.0};
  gt::gtensor<T, 1> h_a(gt::shape(N));
  gt::gtensor<double, 1> h_norm(h_a.shape());

  gt::gtensor_device<T, 1> d_a(h_a.shape());
  gt::gtensor_device<double, 1> d_norm(d_a.shape());

  for (int i = 0; i < N; i++) {
    h_a(i) = T{1., static_cast<double>(i)};
  }

  gt::copy(h_a, d_a);

  gtLaunchKernel(kernel_norm, 1, N, 0, 0, d_a.to_kernel(), d_norm.to_kernel());

  gt::copy(d_norm, h_norm);

  for (int i = 0; i < N; i++) {
    EXPECT_EQ(h_norm(i), gt::norm(h_a(i)));
  }
}

TEST(complex, device_conj)
{
  const int N = 6;
  using T = gt::complex<double>;
  auto I = T{0., 1.0};
  gt::gtensor<T, 1> h_a(gt::shape(N));
  gt::gtensor<T, 1> h_conj(h_a.shape());

  gt::gtensor_device<T, 1> d_a(h_a.shape());
  gt::gtensor_device<T, 1> d_conj(d_a.shape());

  for (int i = 0; i < N; i++) {
    h_a(i) = T{1., static_cast<double>(i)};
  }

  gt::copy(h_a, d_a);

  gtLaunchKernel(kernel_conj, 1, N, 0, 0, d_a.to_kernel(), d_conj.to_kernel());

  gt::copy(d_conj, h_conj);

  for (int i = 0; i < N; i++) {
    EXPECT_EQ(h_conj(i), gt::conj(h_a(i)));
  }
}

template <typename T>
static void run_device_exp(gt::gtensor_device<T, 1>& res,
                           const gt::gtensor_device<T, 1>& x)
{
  auto k_res = res.to_kernel();
  auto k_x = x.to_kernel();

  gt::launch<1>(
    x.shape(), GT_LAMBDA(int i) { k_res(i) = gt::exp(k_x(i)); });
  gt::synchronize();
}

TEST(complex, device_exp_real)
{
  using T = double;

  gt::gtensor_device<T, 1> x = {1.};
  auto res = gt::empty_like(x);
  run_device_exp(res, x);

  gt::gtensor<T, 1> h_res(res.shape());
  copy(res, h_res);
  EXPECT_EQ(h_res(0), std::exp(T(1.)));
}

TEST(complex, device_exp)
{
  using namespace std::complex_literals;
  using T = gt::complex<double>;

  gt::gtensor_device<T, 1> x(gt::shape(1));
  gt::gtensor<T, 1> h_x(x.shape());
  h_x(0) = 1.i * M_PI / 2.;
  copy(h_x, x);

  auto res = gt::empty_like(x);
  run_device_exp(res, x);

  gt::gtensor<T, 1> h_res(res.shape());
  copy(res, h_res);
  EXPECT_LT(gt::abs(h_res(0) - T(0., 1.)), 1e-14);
}

template <typename Tres, typename Tx>
static void run_device_abs(gt::gtensor_device<Tres, 1>& res,
                           const gt::gtensor_device<Tx, 1>& x)
{
  auto k_res = res.to_kernel();
  auto k_x = x.to_kernel();

  gt::launch<1>(
    x.shape(), GT_LAMBDA(int i) { k_res(i) = gt::abs(k_x(i)); });
  gt::synchronize();
}

TEST(complex, device_abs_real)
{
  using T = double;

  gt::gtensor<T, 1> h_x = {-1.75, -0.001};
  gt::gtensor_device<T, 1> x{h_x.shape()};

  gt::copy(h_x, x);

  auto res = gt::empty_like(x);
  run_device_abs(res, x);

  gt::gtensor<T, 1> h_res(res.shape());
  gt::copy(res, h_res);
  gt::synchronize();

  EXPECT_EQ(h_res(0), std::abs(h_x(0)));
  EXPECT_EQ(h_res(1), std::abs(h_x(1)));
}

TEST(complex, device_abs)
{
  using R = double;
  using T = gt::complex<R>;

  gt::gtensor_device<T, 1> x(gt::shape(1));
  gt::gtensor<T, 1> h_x(x.shape());
  h_x(0) = T(sqrt(2.) / 2., sqrt(2.) / 2.);
  gt::copy(h_x, x);

  gt::gtensor_device<R, 1> res(x.shape());
  run_device_abs(res, x);

  gt::gtensor<R, 1> h_res(res.shape());
  gt::copy(res, h_res);
  // EXPECT_EQ(h_res(0), R(1));
}

#endif // CUDA or HIP

#ifdef GTENSOR_DEVICE_SYCL

TEST(complex, sycl_std_cast)
{
  using namespace gt::complex_cast;

  gt::complex<double>* a_complex_double_p;
  auto a_casted = std_cast(a_complex_double_p);
  GT_DEBUG_TYPE(a_casted);
  EXPECT_TRUE((std::is_same<decltype(a_casted), std::complex<double>*>::value));

  double* b_double_p;
  auto b_casted = std_cast(b_double_p);
  GT_DEBUG_TYPE(b_casted);
  EXPECT_TRUE((std::is_same<decltype(b_casted), double*>::value));

  gt::backend::device_ptr<gt::complex<double>> c_complex_double_dp;
  auto c_casted = std_cast(c_complex_double_dp);
  GT_DEBUG_TYPE(c_casted);
  EXPECT_TRUE((std::is_same<decltype(c_casted), std::complex<double>*>::value));

  gt::complex<double>** d_complex_double_pp;
  auto d_casted = std_cast(d_complex_double_pp);
  GT_DEBUG_TYPE(d_casted);
  EXPECT_TRUE(
    (std::is_same<decltype(d_casted), std::complex<double>**>::value));

  const gt::complex<double>* e_complex_double_p_const;
  auto e_casted = std_cast(e_complex_double_p_const);
  GT_DEBUG_TYPE(e_casted);
  EXPECT_TRUE(
    (std::is_same<decltype(e_casted), const std::complex<double>*>::value));
}

TEST(complex, sycl_make_std)
{
  using namespace gt::complex_cast;

  using T1 = gt::complex<double>;
  using T1std = typename make_std<T1>::type;
  GT_DEBUG_TYPE_NAME(T1std);
  EXPECT_TRUE((std::is_same<T1std, std::complex<double>>::value));

  using T2 = float;
  using T2std = typename make_std<float>::type;
  GT_DEBUG_TYPE_NAME(T2std);
  EXPECT_TRUE((std::is_same<T2std, float>::value));
}

#endif // GTENSOR_DEVICE_SYCL

#endif
