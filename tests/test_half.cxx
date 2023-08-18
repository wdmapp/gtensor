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

template <typename S>
void generic_fill_1d(gt::gtensor<gt::half, 1, S>& x, const gt::half& fill_value)
{
  auto k_x = x.to_kernel();

  gt::launch<1, S>(x.shape(), GT_LAMBDA(int i) { k_x(i) = fill_value; });
}

TEST(half, AutoInitHost)
{
    gt::half fill_value{1.25};
    gt::gtensor<gt::half, 1, gt::space::host> a(gt::shape(5), fill_value);
    gt::gtensor<gt::half, 1, gt::space::host> b(a.shape());

    generic_fill_1d<gt::space::host>(b, fill_value);

    EXPECT_EQ(a, b);
}

void host_explicit_haxpy_1d(const gt::half& a,
                            const gt::gtensor<gt::half, 1>& x,
                            gt::gtensor<gt::half, 1>& y)
{
  auto k_x = x.to_kernel();
  auto k_y = y.to_kernel();

  gt::launch_host<1>(
    y.shape(), GT_LAMBDA(int i) { k_y(i) = k_y(i) + a * k_x(i); });
}

TEST(half, HaxpyExplicit1dHost)
{
    gt::gtensor<gt::half, 1, gt::space::host> x(gt::shape(3), 1.5);
    gt::gtensor<gt::half, 1, gt::space::host> y(x.shape(), 2.5);
    gt::half a{0.5};
    gt::gtensor<gt::half, 1, gt::space::host> ref(x.shape(), 3.25);

    host_explicit_haxpy_1d(a, x, y);

    EXPECT_EQ(y, ref);
}

template <typename S>
void generic_explicit_haxpy_1d( const gt::half& a,
                                const gt::gtensor<gt::half, 1, S>& x,
                                      gt::gtensor<gt::half, 1, S>& y)
{
  auto k_x = x.to_kernel();
  auto k_y = y.to_kernel();

  gt::launch<1, S>(
    y.shape(), GT_LAMBDA(int i) { k_y(i) = k_y(i) + a * k_x(i); });
}

TEST(half, HaxpyExplicit1dDevice)
{
    gt::gtensor<gt::half, 1, gt::space::host> h_x(gt::shape(3), gt::half{1.5});
    gt::gtensor<gt::half, 1, gt::space::host> h_y(h_x.shape(), gt::half{2.5});
    gt::half a{0.5};
    gt::gtensor<gt::half, 1, gt::space::host> ref(h_x.shape(), 3.25);

    gt::gtensor<gt::half, 1, gt::space::device> d_x(h_x.shape());
    gt::gtensor<gt::half, 1, gt::space::device> d_y(h_y.shape());
    gt::copy(h_x, d_x);
    gt::copy(h_y, d_y);

    generic_explicit_haxpy_1d<gt::space::device>(a, d_x, d_y);

    gt::copy(d_y, h_y);

    EXPECT_EQ(h_y, ref);
}
