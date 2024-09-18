#include <gtest/gtest.h>

#include <gtensor/gtensor.h>

#include "test_debug.h"

template <typename S>
void generic_double_add_1d(gt::gtensor<double, 1, S>& a,
                           gt::gtensor<double, 1, S>& out)
{
  EXPECT_EQ(a, (gt::gtensor<double, 1, S>{11., 12., 13.}));

  auto k_a = a.to_kernel();
  auto k_out = out.to_kernel();

  gt::launch<1, S>(
    a.shape(), GT_LAMBDA(int i) { k_out(i) = k_a(i) + k_a(i); });
}

void host_double_add_1d(gt::gtensor<double, 1>& a, gt::gtensor<double, 1>& out)
{
  EXPECT_EQ(a, (gt::gtensor<double, 1>{11., 12., 13.}));

  auto k_a = a.to_kernel();
  auto k_out = out.to_kernel();

  gt::launch_host<1>(
    a.shape(), GT_LAMBDA(int i) { k_out(i) = k_a(i) + k_a(i); });
}

TEST(gtensor, launch_1d)
{
  gt::gtensor<double, 1> a{11., 12., 13.};
  gt::gtensor<double, 1> b(a.shape());

  host_double_add_1d(a, b);

  EXPECT_EQ(b, (gt::gtensor<double, 1>{22., 24., 26.}));
}

TEST(gtensor, launch_1d_templated)
{
  gt::gtensor<double, 1> a{11., 12., 13.};
  gt::gtensor<double, 1> b(a.shape());

  generic_double_add_1d<gt::space::host>(a, b);

  EXPECT_EQ(b, (gt::gtensor<double, 1>{22., 24., 26.}));
}

void host_double_view_reverse_1d(gt::gtensor<double, 1>& a,
                                 gt::gtensor<double, 1>& out)
{
  auto k_a = a.to_kernel();
  auto v_out = out.view(gt::slice(2, gt::none, -1));

  EXPECT_EQ(v_out.shape(), gt::shape(3));

  auto k_out = v_out.to_kernel();

  gt::launch_host<1>(
    a.shape(), GT_LAMBDA(int i) { k_out(i) = k_a(i); });
}

TEST(gtensor, launch_view_reverse_1d)
{
  gt::gtensor<double, 1> a{11., 12., 13.};
  gt::gtensor<double, 1> b(a.shape());

  host_double_view_reverse_1d(a, b);

  EXPECT_EQ(b, (gt::gtensor<double, 1>{13., 12., 11.}));
}

#ifdef GTENSOR_HAVE_DEVICE

void device_double_add_1d(gt::gtensor_device<double, 1>& a,
                          gt::gtensor<double, 1>& out)
{
  auto b = gt::empty_like(a);

  auto k_a = a.to_kernel();
  auto k_b = b.to_kernel();

  gt::launch<1>(
    a.shape(), GT_LAMBDA(int i) { k_b(i) = k_a(i) + k_a(i); });
  gt::copy(b, out);
}

TEST(gtensor, device_launch_1d)
{
  gt::gtensor_device<double, 1> a{11., 12., 13.};
  gt::gtensor_device<double, 1> b(a.shape());
  gt::gtensor<double, 1> h_b(a.shape());

  device_double_add_1d(a, h_b);

  EXPECT_EQ(h_b, (gt::gtensor<double, 1>{22., 24., 26.}));

  h_b(0) = 0;
  h_b(1) = 0;
  h_b(2) = 0;
  generic_double_add_1d<gt::space::device>(a, b);
  gt::copy(b, h_b);

  EXPECT_EQ(h_b, (gt::gtensor<double, 1>{22., 24., 26.}));
}

void device_double_view_reverse_1d(gt::gtensor_device<double, 1>& a,
                                   gt::gtensor<double, 1>& out)
{
  auto b = gt::empty_like(a);

  auto k_a = a.to_kernel();
  auto v_b = b.view(gt::slice(2, gt::none, -1));
  auto k_b = v_b.to_kernel();

  gt::launch<1>(
    a.shape(), GT_LAMBDA(int i) { k_b(i) = k_a(i); });

  gt::copy(b, out);
}

TEST(gtensor, device_launch_view_reverse_1d)
{
  gt::gtensor_device<double, 1> a{11., 12., 13.};
  gt::gtensor<double, 1> h_b(a.shape());

  device_double_view_reverse_1d(a, h_b);

  EXPECT_EQ(h_b, (gt::gtensor<double, 1>{13., 12., 11.}));
}

void device_double_add_2d(gt::gtensor_device<double, 2>& a,
                          gt::gtensor<double, 2>& out)
{
  auto b = gt::empty_like(a);

  auto k_a = a.to_kernel();
  auto k_b = b.to_kernel();

  gt::launch<2>(
    a.shape(), GT_LAMBDA(int i, int j) { k_b(i, j) = k_a(i, j) + k_a(i, j); });
  gt::copy(b, out);
}

TEST(gtensor, device_launch_2d)
{
  gt::gtensor_device<double, 2> a{{11., 12., 13.}, {21., 22., 23.}};
  gt::gtensor<double, 2> h_b(a.shape());

  device_double_add_2d(a, h_b);

  EXPECT_EQ(h_b, (gt::gtensor<double, 2>{{22., 24., 26.}, {42., 44., 46.}}));
}

void device_double_add_5d(gt::gtensor_device<double, 5>& a,
                          gt::gtensor<double, 5>& out)
{
  auto b = gt::empty_like(a);

  auto k_a = a.to_kernel();
  auto k_b = b.to_kernel();

  gt::launch<5>(
    a.shape(), GT_LAMBDA(int i, int j, int k, int l, int m) {
      k_b(i, j, k, l, m) = k_a(i, j, k, l, m) + k_a(i, j, k, l, m);
    });
  gt::copy(b, out);
}

void device_double_add_6d(gt::gtensor_device<double, 6>& a,
                          gt::gtensor<double, 6>& out)
{
  auto b = gt::empty_like(a);

  auto k_a = a.to_kernel();
  auto k_b = b.to_kernel();

  gt::launch<6>(
    a.shape(), GT_LAMBDA(int i, int j, int k, int l, int m, int n) {
      k_b(i, j, k, l, m, n) = k_a(i, j, k, l, m, n) + k_a(i, j, k, l, m, n);
    });
  gt::copy(b, out);
}

void device_double_add_7d(gt::gtensor_device<double, 7>& a,
                          gt::gtensor<double, 7>& out)
{
  auto b = gt::empty_like(a);

  auto k_a = a.to_kernel();
  auto k_b = b.to_kernel();

  gt::launch<7>(
    a.shape(), GT_LAMBDA(int i, int j, int k, int l, int m, int n, int o) {
      k_b(i, j, k, l, m, n, o) =
        k_a(i, j, k, l, m, n, o) + k_a(i, j, k, l, m, n, o);
    });
  gt::copy(b, out);
}

TEST(gtensor, device_launch_5d)
{
  gt::gtensor<double, 5> h_a(gt::shape(2, 2, 2, 2, 2));
  gt::gtensor_device<double, 5> a(h_a.shape());
  gt::gtensor<double, 5> h_b(h_a.shape());
  gt::gtensor<double, 5> h_b_expected(h_a.shape());

  for (int i = 0; i < h_a.shape(0); i++) {
    for (int j = 0; j < h_a.shape(1); j++) {
      for (int k = 0; k < h_a.shape(2); k++) {
        for (int l = 0; l < h_a.shape(3); l++) {
          for (int m = 0; m < h_a.shape(4); m++) {
            h_a(i, j, k, l, m) = i + j + k + l + m;
          }
        }
      }
    }
  }

  h_b_expected = 2 * h_a;

  gt::copy(h_a, a);

  device_double_add_5d(a, h_b);

  EXPECT_EQ(h_b, h_b_expected);
}

TEST(gtensor, device_launch_6d)
{
  gt::gtensor<double, 6> h_a(gt::shape(2, 2, 2, 2, 2, 2));
  gt::gtensor_device<double, 6> a(h_a.shape());
  gt::gtensor<double, 6> h_b(h_a.shape());
  gt::gtensor<double, 6> h_b_expected(h_a.shape());

  for (int i = 0; i < h_a.shape(0); i++) {
    for (int j = 0; j < h_a.shape(1); j++) {
      for (int k = 0; k < h_a.shape(2); k++) {
        for (int l = 0; l < h_a.shape(3); l++) {
          for (int m = 0; m < h_a.shape(4); m++) {
            for (int n = 0; n < h_a.shape(5); n++) {
              h_a(i, j, k, l, m, n) = i + j + k + l + m + n;
            }
          }
        }
      }
    }
  }

  h_b_expected = 2 * h_a;

  gt::copy(h_a, a);

  device_double_add_6d(a, h_b);

  EXPECT_EQ(h_b, h_b_expected);
}

TEST(gtensor, device_launch_7d)
{
  gt::gtensor<double, 7> h_a(gt::shape(2, 2, 2, 2, 2, 2, 2));
  gt::gtensor_device<double, 7> a(h_a.shape());
  gt::gtensor<double, 7> h_b(h_a.shape());
  gt::gtensor<double, 7> h_b_expected(h_a.shape());

  for (int i = 0; i < h_a.shape(0); i++) {
    for (int j = 0; j < h_a.shape(1); j++) {
      for (int k = 0; k < h_a.shape(2); k++) {
        for (int l = 0; l < h_a.shape(3); l++) {
          for (int m = 0; m < h_a.shape(4); m++) {
            for (int n = 0; n < h_a.shape(5); n++) {
              for (int o = 0; o < h_a.shape(6); o++) {
                h_a(i, j, k, l, m, n, o) = i + j + k + l + m + n + o;
              }
            }
          }
        }
      }
    }
  }

  h_b_expected = 2 * h_a;

  gt::copy(h_a, a);

  device_double_add_7d(a, h_b);

  EXPECT_EQ(h_b, h_b_expected);
}

#endif // GTENSOR_HAVE_DEVICE
