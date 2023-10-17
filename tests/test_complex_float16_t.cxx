#include <gtest/gtest.h>

#include <gtensor/gtensor.h>

#include <gtensor/complex_float16_t.h>
#include <gtensor/float16_t.h>

TEST(complex_float16_t, comparison_operators)
{
  gt::complex_float16_t a{7.0, -2.0};

  EXPECT_EQ(a, a);
}
