#include <gtest/gtest.h>

#include <gtensor/gtensor.h>

#include <gtensor/half.h>

TEST(half, ScalarArithmetic)
{
    gt::half a{1.0};
    gt::half b{2.0};

    gt::half c{0.0};
    gt::half ref{0.0};

    c = a + b;
    ref = 3.0;
    EXPECT_EQ(c, ref);

    c = a - b;
    ref = -1.0;
    EXPECT_EQ(c, ref);

    c = a * b;
    ref = 2.0;
    EXPECT_EQ(c, ref);

    c = a / b;
    ref = 0.5;
    EXPECT_EQ(c, ref);
}
