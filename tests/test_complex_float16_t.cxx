#include <gtest/gtest.h>

#include <gtensor/gtensor.h>

#include <gtensor/complex_float16_t.h>
#include <gtensor/float16_t.h>

#include <sstream>

TEST(complex_float16_t, comparison_operators)
{
  gt::complex_float16_t a{7.0, -2.0};
  gt::complex_float16_t b{6.0, -3.0};
  gt::complex_float16_t c{7.0, -3.0};
  gt::complex_float16_t d{6.0, -2.0};

  EXPECT_EQ(a, a);
  EXPECT_NE(a, b);
  EXPECT_NE(a, c);
  EXPECT_NE(a, d);

  gt::complex_float16_t e{3.0, 0.0};
  gt::complex_float16_t f{3.0, 1.0};
  gt::float16_t s{3.0};
  gt::float16_t t{4.0};

  EXPECT_EQ(e, s);
  EXPECT_EQ(s, e);
  EXPECT_NE(f, s);
  EXPECT_NE(s, f);
  EXPECT_NE(e, t);
  EXPECT_NE(t, e);
  EXPECT_NE(f, t);
  EXPECT_NE(t, f);
}

TEST(complex_float16_t, constructors)
{
  gt::complex_float16_t a{7.0, -2.0};
  gt::complex_float16_t b{a};
  gt::complex<float> c{7.0, -2.0};
  gt::complex_float16_t d{c};

  EXPECT_EQ(a, b);
  EXPECT_EQ(a, d);
}

TEST(complex_float16_t, assignment)
{
  gt::complex_float16_t a{7.0, -2.0};
  gt::complex_float16_t b{0.0, 0.0};
  gt::complex_float16_t c{3.0, 0.0};
  gt::float16_t x{3.0};
  gt::complex_float16_t e{2.0, 1.0};
  gt::complex<float> f{2.0, 1.0};

  b = a;
  EXPECT_EQ(a, b);

  b = x;
  EXPECT_EQ(c, b);

  b = f;
  EXPECT_EQ(e, b);
}

TEST(complex_float16_t, getter_setter)
{
  const gt::complex_float16_t a{7.0, -2.0};
  gt::complex_float16_t b;

  b.real(a.real());
  b.imag(a.imag());

  EXPECT_EQ(a, b);
}

TEST(complex_float16_t, update_operators)
{
  gt::complex_float16_t a{5.0, -3.0};
  gt::complex_float16_t b{-2.0, 2.0};
  gt::complex<float> f{-2.0, 2.0};
  gt::complex_float16_t ref;
  gt::float16_t x{3.0};

  a += b;
  ref = gt::complex_float16_t{3.0, -1.0};
  EXPECT_EQ(a, ref);

  a += x;
  ref = gt::complex_float16_t{6.0, -1.0};
  EXPECT_EQ(a, ref);

  a -= b;
  ref = gt::complex_float16_t{8.0, -3.0};
  EXPECT_EQ(a, ref);

  a -= x;
  ref = gt::complex_float16_t{5.0, -3.0};
  EXPECT_EQ(a, ref);

  a *= b;
  ref = gt::complex_float16_t{-4.0, 16.0};
  EXPECT_EQ(a, ref);

  a *= x;
  ref = gt::complex_float16_t{-12.0, 48.0};
  EXPECT_EQ(a, ref);

  a /= x;
  ref = gt::complex_float16_t{-4.0, 16.0};
  EXPECT_EQ(a, ref);

  a /= b;
  ref = gt::complex_float16_t{5.0, -3.0};
  EXPECT_EQ(a, ref); // exact because b chosen s.t. norm(b) = 8

  a += f;
  ref = gt::complex_float16_t{3.0, -1.0};
  EXPECT_EQ(a, ref);

  a += x;
  a -= f;
  ref = gt::complex_float16_t{8.0, -3.0};
  EXPECT_EQ(a, ref);

  a -= x;
  a *= f;
  ref = gt::complex_float16_t{-4.0, 16.0};
  EXPECT_EQ(a, ref);

  a /= f;
  ref = gt::complex_float16_t{5.0, -3.0};
  EXPECT_EQ(a, ref); // exact because f chosen s.t. norm(b) = 8
}

TEST(complex_float16_t, values)
{
  gt::complex_float16_t a{4.0, -3.0};

  gt::float16_t a_real{4.0};
  gt::float16_t a_imag{-3.0};
  gt::float16_t a_abs{5.0};
  gt::float16_t a_norm{25.0};
  gt::complex_float16_t a_conj{4.0, +3.0};

  EXPECT_EQ(a_real, real(a));
  EXPECT_EQ(a_imag, imag(a));
  EXPECT_EQ(a_abs, abs(a));
  EXPECT_EQ(a_norm, norm(a));
  EXPECT_EQ(a_conj, conj(a));
}

TEST(complex_float16_t, binary_arithmetic_operators)
{
  gt::complex_float16_t a{4.0, -4.0};
  gt::complex_float16_t b{-2.0, 2.0};
  gt::float16_t x{8.0};
  gt::complex_float16_t c;
  gt::complex_float16_t ref;

  c = a + b;
  ref = gt::complex_float16_t{2.0, -2.0};
  EXPECT_EQ(c, ref);
  c = a + x;
  ref = gt::complex_float16_t{12.0, -4.0};
  EXPECT_EQ(c, ref);
  c = x + a;
  EXPECT_EQ(c, ref);

  c = a - b;
  ref = gt::complex_float16_t{6.0, -6.0};
  EXPECT_EQ(c, ref);
  c = a - x;
  ref = gt::complex_float16_t{-4.0, -4.0};
  EXPECT_EQ(c, ref);
  c = x - a;
  ref = gt::complex_float16_t{4.0, 4.0};
  EXPECT_EQ(c, ref);

  c = a * b;
  ref = gt::complex_float16_t{0.0, 16.0};
  EXPECT_EQ(c, ref);
  c = a * x;
  ref = gt::complex_float16_t{32.0, -32.0};
  EXPECT_EQ(c, ref);
  c = x * a;
  EXPECT_EQ(c, ref);

  c = a / b;
  ref = gt::complex_float16_t{-2.0, 0.0};
  EXPECT_EQ(c, ref); // exact because b chosen s.t. norm(b) = 8
  c = a / x;
  ref = gt::complex_float16_t{0.5, -0.5};
  EXPECT_EQ(c, ref);
  ref = gt::complex_float16_t{1.0, 1.0};
  c = x / a;
  EXPECT_EQ(c, ref); // exact because a chosen s.t. norm(a) = 32
}

TEST(complex_float16_t, unary_arithmetic_operators)
{
  gt::complex_float16_t a{4.0, -5.0};
  gt::complex_float16_t b{-4.0, 5.0};
  gt::complex_float16_t c;

  c = +a;
  EXPECT_EQ(c, a);

  c = -a;
  EXPECT_EQ(c, b);
}

TEST(complex_float16_t, iostream)
{
  std::istringstream is("(1.125, -2.5)");
  std::ostringstream os;
  gt::complex_float16_t a{1.125, -2.5};
  gt::complex_float16_t b;

  is >> b;
  EXPECT_EQ(a, b);

  os << a;
  EXPECT_EQ(is.str(), os.str());
}

#ifdef GTENSOR_HAVE_DEVICE

TEST(complex_float16_t, device_complex_ops)
{
  using T = gt::complex_float16_t;
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
TEST(complex_float16_t, device_float16_t_multiply)
{
  using T = gt::float16_t;
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
TEST(complex_float16_t, device_complex_multiply)
{
  using T = gt::complex_float16_t;
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
TEST(complex_float16_t, device_eval)
{
  using T = gt::complex_float16_t;
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
  auto e = T{1. / 2.} * (e1 + e2);
  std::cout << "e   type: " << typeid(e).name() << "\n";
  auto c = eval(e);
  std::cout << "c   type: " << typeid(c).name() << std::endl;

  gt::copy(c, h_c);

  EXPECT_EQ(h_c, h_a);
}

#if defined(GTENSOR_DEVICE_CUDA) || defined(GTENSOR_DEVICE_HIP)

__global__ void kernel_norm(
  gt::gtensor_span_device<gt::complex_float16_t, 1> d_in,
  gt::gtensor_span_device<gt::float16_t, 1> d_out)
{
  int i = threadIdx.x;
  if (i < d_in.shape(0)) {
    d_out(i) = gt::norm(d_in(i));
  }
}

__global__ void kernel_conj(
  gt::gtensor_span_device<gt::complex_float16_t, 1> d_in,
  gt::gtensor_span_device<gt::complex_float16_t, 1> d_out)
{
  int i = threadIdx.x;
  if (i < d_in.shape(0)) {
    d_out(i) = gt::conj(d_in(i));
  }
}

TEST(complex_float16_t, device_norm)
{
  const int N = 6;
  using T = gt::complex_float16_t;
  auto I = T{0., 1.0};
  gt::gtensor<T, 1> h_a(gt::shape(N));
  gt::gtensor<gt::float16_t, 1> h_norm(h_a.shape());

  gt::gtensor_device<T, 1> d_a(h_a.shape());
  gt::gtensor_device<gt::float16_t, 1> d_norm(d_a.shape());

  for (int i = 0; i < N; i++) {
    h_a(i) = T{1., static_cast<gt::float16_t>(i)};
  }

  gt::copy(h_a, d_a);

  gtLaunchKernel(kernel_norm, 1, N, 0, 0, d_a.to_kernel(), d_norm.to_kernel());

  gt::copy(d_norm, h_norm);

  for (int i = 0; i < N; i++) {
    EXPECT_EQ(h_norm(i), gt::norm(h_a(i)));
  }
}

TEST(complex_float16_t, device_conj)
{
  const int N = 6;
  using T = gt::complex_float16_t;
  auto I = T{0., 1.0};
  gt::gtensor<T, 1> h_a(gt::shape(N));
  gt::gtensor<T, 1> h_conj(h_a.shape());

  gt::gtensor_device<T, 1> d_a(h_a.shape());
  gt::gtensor_device<T, 1> d_conj(d_a.shape());

  for (int i = 0; i < N; i++) {
    h_a(i) = T{1., static_cast<gt::float16_t>(i)};
  }

  gt::copy(h_a, d_a);

  gtLaunchKernel(kernel_conj, 1, N, 0, 0, d_a.to_kernel(), d_conj.to_kernel());

  gt::copy(d_conj, h_conj);

  for (int i = 0; i < N; i++) {
    EXPECT_EQ(h_conj(i), gt::conj(h_a(i)));
  }
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

TEST(complex_float16_t, device_abs_real)
{
  using T = gt::float16_t;

  gt::gtensor<T, 1> h_x = {-1.75, -0.001};
  gt::gtensor_device<T, 1> x{h_x.shape()};

  gt::copy(h_x, x);

  auto res = gt::empty_like(x);
  run_device_abs(res, x);

  gt::gtensor<T, 1> h_res(res.shape());
  gt::copy(res, h_res);
  gt::synchronize();

  EXPECT_EQ(h_res(0), gt::abs(h_x(0)));
  EXPECT_EQ(h_res(1), gt::abs(h_x(1)));
}

TEST(complex_float16_t, device_abs)
{
  using R = gt::float16_t;
  using T = gt::complex_float16_t;

  gt::gtensor_device<T, 1> x(gt::shape(1));
  gt::gtensor<T, 1> h_x(x.shape());
  h_x(0) = T(sqrt(2.) / 2., sqrt(2.) / 2.);
  gt::copy(h_x, x);

  gt::gtensor_device<R, 1> res(x.shape());
  run_device_abs(res, x);

  gt::gtensor<R, 1> h_res(res.shape());
  gt::copy(res, h_res);
  // here, truncation and rounding errors cancel for IEEE binary16
  EXPECT_EQ(h_res(0), R(1));
}

#endif // CUDA or HIP

#endif // GTENSOR_HAVE_DEVICE
