#include <gtest/gtest.h>

#include <gtensor/gtensor.h>

#include <gtensor/half.h>

template <typename S, typename fp_type>
void generic_fill_1d(gt::gtensor<fp_type, 1, S>& x, const fp_type& fill_value)
{
  auto k_x = x.to_kernel();

  gt::launch<1, S>(
    x.shape(), GT_LAMBDA(int i) { k_x(i) = fill_value; });
}

TEST(halfFailing, AutoInitHostFloat)
{
    float fill_value{1.25};
    gt::gtensor<float, 1, gt::space::host> a(gt::shape(5), fill_value);
    gt::gtensor<float, 1, gt::space::host> b(a.shape());

    generic_fill_1d<gt::space::host>(b, fill_value);

    EXPECT_EQ(a, b);
}

TEST(halfFailing, AutoInitHostHalf)
{
    gt::half fill_value{1.25};
    gt::gtensor<gt::half, 1, gt::space::host> a(gt::shape(5), fill_value);
    gt::gtensor<gt::half, 1, gt::space::host> b(a.shape());

    generic_fill_1d<gt::space::host>(b, fill_value);

    EXPECT_EQ(a, b);
}

TEST(halfFailing, AutoInitDeviceFloat)
{
    float fill_value{1.25};
    gt::gtensor<float, 1, gt::space::device> d_a(gt::shape(5), fill_value);
    gt::gtensor<float, 1, gt::space::device> d_b(d_a.shape());

    generic_fill_1d<gt::space::device>(d_b, fill_value);

    gt::gtensor<float, 1, gt::space::host> h_a(d_a.shape());
    gt::copy(d_a, h_a);
    gt::gtensor<float, 1, gt::space::host> h_b(d_a.shape());
    gt::copy(d_b, h_b);

    EXPECT_EQ(h_a, h_b);
}

TEST(halfFailing, AutoInitDeviceHalf)
{
    gt::half fill_value{1.25};
    // DOES NOT COMPILE !!! ----------------------------------------------------
    // gt::gtensor<gt::half, 1, gt::space::device> d_a(gt::shape(5), fill_value);
    EXPECT_EQ(true, false);
    // temporary workaround:
    gt::gtensor<gt::half, 1, gt::space::device> d_a(gt::shape(5));
    generic_fill_1d<gt::space::device>(d_a, fill_value);
    // -------------------------------------------------------------------------

    gt::gtensor<gt::half, 1, gt::space::device> d_b(d_a.shape());

    generic_fill_1d<gt::space::device>(d_b, fill_value);

    gt::gtensor<gt::half, 1, gt::space::host> h_a(d_a.shape());
    gt::copy(d_a, h_a);
    gt::gtensor<gt::half, 1, gt::space::host> h_b(d_a.shape());
    gt::copy(d_b, h_b);

    EXPECT_EQ(h_a, h_b);
}
