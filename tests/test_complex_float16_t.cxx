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

