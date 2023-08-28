#include <gtest/gtest.h>

#include <gtensor/gtensor.h>

#include <gtensor/half.h>

TEST(half, scalar_arithmetic)
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

TEST(half, binary_comparison_operators)
{
    gt::half a{1.0};
    gt::half b{2.0};
    gt::half c{2.0};

    EXPECT_EQ(a, a);
    EXPECT_EQ(b, b);
    EXPECT_EQ(b, c);
    EXPECT_EQ(c, b);
    EXPECT_EQ(c, c);

    EXPECT_NE(a, b);
    EXPECT_NE(a, c);
    EXPECT_NE(b, a);
    EXPECT_NE(c, a);

    EXPECT_LT(a, b);
    EXPECT_LT(a, c);

    EXPECT_LE(a, a);
    EXPECT_LE(a, b);
    EXPECT_LE(a, c);
    EXPECT_LE(b, b);
    EXPECT_LE(b, c);
    EXPECT_LE(c, b);
    EXPECT_LE(c, c);

    EXPECT_GT(b, a);
    EXPECT_GT(c, a);

    EXPECT_GE(a, a);
    EXPECT_GE(b, a);
    EXPECT_GE(b, b);
    EXPECT_GE(b, c);
    EXPECT_GE(c, a);
    EXPECT_GE(c, b);
    EXPECT_GE(c, c);
}

template <typename S>
void generic_fill_1d(gt::gtensor<gt::half, 1, S>& x, const gt::half& fill_value)
{
    auto k_x = x.to_kernel();

    gt::launch<1, S>(x.shape(), GT_LAMBDA(int i) { k_x(i) = fill_value; });
}

TEST(half, auto_init_host)
{
    gt::half fill_value{1.25};
    gt::gtensor<gt::half, 1, gt::space::host> a(gt::shape(5), fill_value);
    gt::gtensor<gt::half, 1, gt::space::host> b(a.shape());

    generic_fill_1d<gt::space::host>(b, fill_value);

    EXPECT_EQ(a, b);
}

TEST(half, auto_init_device)
{
    gt::half fill_value{1.25};
    gt::gtensor<gt::half, 1, gt::space::device> a(gt::shape(5), fill_value);
    gt::gtensor<gt::half, 1, gt::space::device> b(a.shape());

    generic_fill_1d<gt::space::device>(b, fill_value);

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

TEST(half, haxpy_explicit_1d_host)
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

TEST(half, haxpy_explicit_1d_device)
{
    gt::gtensor<gt::half, 1, gt::space::device> x(gt::shape(3), 1.5);
    gt::gtensor<gt::half, 1, gt::space::device> y(x.shape(), 2.5);
    gt::half a{0.5};
    gt::gtensor<gt::half, 1, gt::space::device> ref(y.shape(), 3.25);

    generic_explicit_haxpy_1d<gt::space::device>(a, x, y);

    EXPECT_EQ(y, ref);
}

TEST(half, haxpy_implicit_1d_host)
{
    gt::gtensor<gt::half, 1, gt::space::host> x(gt::shape(3), 1.5);
    gt::gtensor<gt::half, 1, gt::space::host> y(x.shape(), 2.5);
    gt::half a{0.5};
    gt::gtensor<gt::half, 1, gt::space::host> ref(x.shape(), 3.25);

    y = a * x + y;

    EXPECT_EQ(y, ref);
}

TEST(half, haxpy_implicit_1d_device)
{
    gt::gtensor<gt::half, 1, gt::space::device> x(gt::shape(3), 1.5);
    gt::gtensor<gt::half, 1, gt::space::device> y(x.shape(), 2.5);
    gt::half a{0.5};
    gt::gtensor<gt::half, 1, gt::space::device> ref(y.shape(), 3.25);

    y = a * x + y;

    EXPECT_EQ(y, ref);
}

template <typename S>
void generic_explicit_custom_kernel_1d( const gt::half& s1,
                                        const gt::half& s2,
                                        const gt::gtensor<gt::half, 1, S>& a,
                                        const gt::gtensor<gt::half, 1, S>& b,
                                        const gt::gtensor<gt::half, 1, S>& c,
                                        const gt::gtensor<gt::half, 1, S>& d,
                                        const gt::gtensor<gt::half, 1, S>& e,
                                        gt::gtensor<gt::half, 1, S>& result)
{
    auto k_a = a.to_kernel();
    auto k_b = b.to_kernel();
    auto k_c = c.to_kernel();
    auto k_d = d.to_kernel();
    auto k_e = e.to_kernel();
    auto k_r = result.to_kernel();

    gt::launch<1, S>(result.shape(), GT_LAMBDA(int i)
        { k_r(i) = s2 - k_e(i) * ((k_a(i) - s1 * k_b(i)) / k_c(i) + k_d(i)); });
}

TEST(half, custom_kernel_explicit_implicit_host_device)
{
    gt::half a_val{12.34}, b_val{2.345}, c_val{0.987}, d_val{0.67}, e_val{3.14};
    gt::half s1{0.1}, s2{4.56};

    gt::half r = s2 - e_val * ((a_val - s1 * b_val) / c_val + d_val);

    auto shape = gt::shape(3);

    gt::gtensor<gt::half, 1, gt::space::host> h_a(shape, a_val);
    gt::gtensor<gt::half, 1, gt::space::host> h_b(shape, b_val);
    gt::gtensor<gt::half, 1, gt::space::host> h_c(shape, c_val);
    gt::gtensor<gt::half, 1, gt::space::host> h_d(shape, d_val);
    gt::gtensor<gt::half, 1, gt::space::host> h_e(shape, e_val);
    gt::gtensor<gt::half, 1, gt::space::host> h_r_expl(shape);
    gt::gtensor<gt::half, 1, gt::space::host> h_r_impl(shape);

    gt::gtensor<gt::half, 1, gt::space::device> d_a(shape, a_val);
    gt::gtensor<gt::half, 1, gt::space::device> d_b(shape, b_val);
    gt::gtensor<gt::half, 1, gt::space::device> d_c(shape, c_val);
    gt::gtensor<gt::half, 1, gt::space::device> d_d(shape, d_val);
    gt::gtensor<gt::half, 1, gt::space::device> d_e(shape, e_val);
    gt::gtensor<gt::half, 1, gt::space::device> d_r_expl(shape);
    gt::gtensor<gt::half, 1, gt::space::device> d_r_impl(shape);

    h_r_impl = s2 - h_e * ((h_a - s1 * h_b) / h_c + h_d);
    d_r_impl = s2 - d_e * ((d_a - s1 * d_b) / d_c + d_d);

    generic_explicit_custom_kernel_1d<gt::space::host>(s1, s2,
            h_a, h_b, h_c, h_d, h_e, h_r_expl);

    generic_explicit_custom_kernel_1d<gt::space::device>(s1, s2,
            d_a, d_b, d_c, d_d, d_e, d_r_expl);

    EXPECT_EQ(h_r_impl(2), r);
    EXPECT_EQ(h_r_impl, h_r_expl);
    EXPECT_EQ(h_r_impl, d_r_expl);
    EXPECT_EQ(h_r_impl, d_r_impl);
}

TEST(half, mixed_precision_scalar)
{
    gt::half a_half{1.0};

    gt::half b_half{2.0};
    float b_float{2.0};
    double b_double{2.0};

    auto c_half = a_half + b_half;
    auto c_float = a_half + b_float;
    auto c_double = a_half + b_double;

    EXPECT_TRUE((std::is_same<gt::half, decltype(c_half)>::value));
    EXPECT_TRUE((std::is_same<float, decltype(c_float)>::value));
    EXPECT_TRUE((std::is_same<double, decltype(c_double)>::value));

    EXPECT_EQ(c_half, c_float);
    EXPECT_EQ(c_half, c_double);
}

TEST(half, mixed_precision_host)
{
    auto shape = gt::shape(3);
    gt::gtensor<gt::half, 1, gt::space::host> vh(shape, 4.0);
    gt::gtensor<float, 1, gt::space::host> vf(shape, 3.0);
    gt::gtensor<double, 1, gt::space::host> vd(shape, 2.0);

    gt::gtensor<gt::half, 1, gt::space::host> rh(shape);
    gt::gtensor<float, 1, gt::space::host> rf(shape);
    gt::gtensor<double, 1, gt::space::host> rd(shape);

    gt::gtensor<double, 1, gt::space::host> ref(shape, 10.0);

    rh = (vh * vf) - (vh / vd);
    rf = (vh * vf) - (vh / vd);
    rd = (vh * vf) - (vh / vd);

    EXPECT_EQ(ref, rh);
    EXPECT_EQ(ref, rf);
    EXPECT_EQ(ref, rd);
}

TEST(half, mixed_precision_device)
{
    auto shape = gt::shape(3);
    gt::gtensor<gt::half, 1, gt::space::device> vh(shape, 4.0);
    gt::gtensor<float, 1, gt::space::device> vf(shape, 3.0);
    gt::gtensor<double, 1, gt::space::device> vd(shape, 2.0);

    gt::gtensor<gt::half, 1, gt::space::device> rh(shape);
    gt::gtensor<float, 1, gt::space::device> rf(shape);
    gt::gtensor<double, 1, gt::space::device> rd(shape);

    gt::gtensor<double, 1, gt::space::device> ref(shape, 10.0);

    rh = (vh * vf) - (vh / vd);
    rf = (vh * vf) - (vh / vd);
    rd = (vh * vf) - (vh / vd);

    EXPECT_EQ(ref, rh);
    EXPECT_EQ(ref, rf);
    EXPECT_EQ(ref, rd);
}
