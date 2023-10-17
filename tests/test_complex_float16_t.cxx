#include <gtest/gtest.h>

#include <gtensor/gtensor.h>

#include <gtensor/complex_float16_t.h>
#include <gtensor/float16_t.h>

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

}
