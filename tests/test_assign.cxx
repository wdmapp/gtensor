#include <gtest/gtest.h>

#include <iostream>
#include <stdexcept>

#include "gtensor/gtensor.h"

#include "test_debug.h"

TEST(assign, gtensor_6d)
{
  gt::gtensor<int, 6> a(gt::shape(2, 3, 4, 5, 6, 7));
  gt::gtensor<int, 6> b(a.shape());

  int* adata = a.data();

  for (int i = 0; i < a.size(); i++) {
    adata[i] = i;
  }

  EXPECT_NE(a, b);
  b = a;
  EXPECT_EQ(a, b);
}

TEST(assign, gview_1d_scalar)
{
  auto a = gt::empty<int>(gt::shape(5));
  auto aview = a.view(gt::all);

  aview = gt::scalar(5);

  EXPECT_EQ(a, (gt::gtensor<int, 1>{5, 5, 5, 5, 5}));
}

TEST(assign, gtensor_fill)
{
  gt::gtensor<float, 2> a(gt::shape(2, 3));

  a.fill(9001);

  EXPECT_EQ(a,
            (gt::gtensor<float, 2>{{9001, 9001}, {9001, 9001}, {9001, 9001}}));
}

TEST(assign, gview_fill)
{
  gt::gtensor<float, 2> a(gt::shape(2, 3));
  auto av = a.view(gt::all, gt::all);

  av.fill(9001);

  EXPECT_EQ(a,
            (gt::gtensor<float, 2>{{9001, 9001}, {9001, 9001}, {9001, 9001}}));
}

TEST(assign, span_fill)
{
  gt::gtensor<float, 2> a(gt::shape(2, 3));
  auto as = a.to_kernel();

  as.fill(9001);

  EXPECT_EQ(a,
            (gt::gtensor<float, 2>{{9001, 9001}, {9001, 9001}, {9001, 9001}}));
}

TEST(assign, broadcast_6d)
{
  gt::gtensor<int, 6> a(gt::shape(8, 1, 2, 4, 1, 1), 0);
  gt::gtensor<int, 6> b(gt::shape(8, 1, 2, 1, 1, 1), -7);

  gt::assign(a, b);

  for (int i = 0; i < a.shape(0); i++) {
    for (int j = 0; j < a.shape(2); j++) {
      for (int k = 0; k < a.shape(3); k++) {
        EXPECT_EQ(a(i, 0, j, k, 0, 0), -7);
      }
    }
  }
}

#ifdef GTENSOR_HAVE_DEVICE

TEST(assign, device_gtensor_6d)
{
  gt::gtensor_device<int, 6> a(gt::shape(2, 3, 4, 5, 6, 7));
  gt::gtensor_device<int, 6> b(a.shape());
  gt::gtensor<int, 6> h_a(a.shape());
  gt::gtensor<int, 6> h_b(a.shape());

  int* adata = h_a.data();

  for (int i = 0; i < a.size(); i++) {
    adata[i] = i;
  }

  gt::copy(h_a, a);
  b = a;
  gt::copy(b, h_b);

  EXPECT_EQ(h_a, h_b);
}

TEST(assign, device_gtensor_fill)
{
  gt::gtensor_device<float, 2> a(gt::shape(2, 3));
  gt::gtensor<float, 2> h_a(a.shape());

  a.fill(9001);

  gt::copy(a, h_a);
  EXPECT_EQ(h_a, (gt::gtensor_device<float, 2>{
                   {9001, 9001}, {9001, 9001}, {9001, 9001}}));
}

TEST(assign, device_gview_fill)
{
  gt::gtensor_device<float, 2> a(gt::shape(2, 3));
  gt::gtensor<float, 2> h_a(a.shape());
  auto av = a.view(gt::all, gt::all);

  av.fill(9001);

  gt::copy(a, h_a);
  EXPECT_EQ(h_a,
            (gt::gtensor<float, 2>{{9001, 9001}, {9001, 9001}, {9001, 9001}}));
}

TEST(assign, device_span_fill)
{
  gt::gtensor_device<float, 2> a(gt::shape(2, 3));
  gt::gtensor<float, 2> h_a(a.shape());
  auto as = a.to_kernel();

  as.fill(9001);

  gt::copy(a, h_a);
  EXPECT_EQ(h_a,
            (gt::gtensor<float, 2>{{9001, 9001}, {9001, 9001}, {9001, 9001}}));
}

TEST(assign, device_gview_1d_scalar)
{
  auto a = gt::empty_device<int>(gt::shape(5));
  auto h_a = gt::empty_like(a);
  auto aview = a.view(gt::all);

  aview = gt::scalar(5);

  gt::copy(a, h_a);

  EXPECT_EQ(h_a, (gt::gtensor<int, 1>{5, 5, 5, 5, 5}));
}

TEST(assign, device_gtensor_large_2d)
{
  gt::gtensor_device<int, 2> a(gt::shape(2, 17920000));
  gt::gtensor_device<int, 2> b(a.shape());
  gt::gtensor<int, 2> h_a(a.shape());
  gt::gtensor<int, 2> h_b(a.shape());

  int* adata = h_a.data();

  for (int i = 0; i < a.size(); i++) {
    adata[i] = i;
  }

  gt::copy(h_a, a);
  // NB: b = a calls the default operator= which ends up triggering
  // and underlying storage vector copy, usually a device memcpy, so
  // it doesn't launch the gtensor assign kernel. Call assign directly
  // to exercise the code
  assign(b, a);
  gt::copy(b, h_b);

  EXPECT_EQ(h_a, h_b);
}

TEST(assign, device_view_noncontiguous_6d)
{
  using T = gt::complex<double>;

  int nzb = 2;
  int nvb = 2;
  int nwb = 2;

  // ijklmn, no ghost
  auto g_shape = gt::shape(32, 4, 48, 40, 30, 2);

  // ijklmn, ghost in z, v, w
  auto f_shape =
    gt::shape(g_shape[0], g_shape[1], g_shape[2] + 2 * nzb,
              g_shape[3] + 2 * nvb, g_shape[4] + 2 * nwb, g_shape[5]);
  // i klmn, no ghost
  auto papbar_shape =
    gt::shape(g_shape[0], g_shape[2], g_shape[3], g_shape[4], g_shape[5]);
  // ijz mn, ghost in z
  auto bar_apar_shape =
    gt::shape(g_shape[0], g_shape[1], f_shape[2], g_shape[4], g_shape[5]);
  auto h_g = gt::full(g_shape, T(2.0));
  auto d_g = gt::empty_device<T>(g_shape);
  auto h_papbar = gt::full(papbar_shape, T(1.5));
  auto d_papbar = gt::empty_device<T>(papbar_shape);
  auto h_bar_apar = gt::full(bar_apar_shape, T(0.0, -1.0));
  auto d_bar_apar = gt::empty_device<T>(bar_apar_shape);
  auto h_f = gt::full<T>(f_shape, T(100.0));
  auto d_f = gt::empty_device<T>(f_shape);

  gt::copy(h_g, d_g);
  gt::copy(h_papbar, d_papbar);
  gt::copy(h_bar_apar, d_bar_apar);
  gt::copy(h_f, d_f);

  auto lhs_view = d_f.view(gt::all, gt::all, gt::slice(nzb, -nzb),
                           gt::slice(nvb, -nvb), gt::slice(nwb, -nwb), gt::all);
  auto d_papbar_view =
    d_papbar.view(gt::all, gt::newaxis, gt::all, gt::all, gt::all, gt::all);
  auto d_bar_apar_view = d_bar_apar.view(gt::all, gt::all, gt::slice(nzb, -nzb),
                                         gt::newaxis, gt::all, gt::all);
  auto rhs_view = d_g + d_papbar_view * d_bar_apar_view;

  GT_DEBUG_VAR(d_g.shape());
  GT_DEBUG_VAR(d_papbar_view.shape());
  GT_DEBUG_VAR(d_bar_apar_view.shape());
  GT_DEBUG_VAR(lhs_view.shape());
  GT_DEBUG_VAR(rhs_view.shape());

  lhs_view = rhs_view;

  /*
  d_f.view(gt::all, gt::all, gt::slice(nzb, -nzb), gt::slice(nvb, -nvb),
  gt::slice(nwb, -nwb), gt::all) = d_g + d_papbar.view(gt::all, gt::newaxis,
  gt::all, gt::all, gt::all, gt::all) * d_bar_apar.view(gt::all, gt::all,
  gt::slice(nzb, -nzb), gt::newaxis, gt::all, gt::all);
  */

  gt::copy(d_f, h_f);

  // spot check boundary, not changed
  EXPECT_EQ(h_f(3, 3, nzb - 1, nvb, nwb, 1), T(100.0));
  EXPECT_EQ(h_f(3, 3, nzb, nvb - 1, nwb, 1), T(100.0));
  EXPECT_EQ(h_f(3, 3, nzb, nvb, nwb - 1, 1), T(100.0));
  EXPECT_EQ(
    h_f(0, 1, f_shape[2] - nzb, f_shape[3] - nvb - 1, f_shape[4] - nwb - 1, 1),
    T(100.0));
  EXPECT_EQ(
    h_f(0, 1, f_shape[2] - nzb - 1, f_shape[3] - nvb, f_shape[4] - nwb - 1, 1),
    T(100.0));
  EXPECT_EQ(
    h_f(0, 1, f_shape[2] - nzb - 1, f_shape[3] - nvb - 1, f_shape[4] - nwb, 1),
    T(100.0));

  // spot check inside that was changed
  EXPECT_EQ(h_f(0, 1, f_shape[2] - nzb - 1, f_shape[3] - nvb - 1,
                f_shape[4] - nwb - 1, 1),
            T(2.0, -1.5));
  EXPECT_EQ(h_f(3, 3, nzb, nvb, nwb, 1), T(2.0, -1.5));

  gt::synchronize();
}

TEST(assign, device_view_noncontiguous_6d_scalar)
{
  using T = gt::complex<double>;

  int nzb = 2;
  int nvb = 2;
  int nwb = 0;

  // ijklmn, ghost in z, v, w
  auto f_shape = gt::shape(5, 7, 9, 11, 13, 2);

  auto h_f = gt::full<T>(f_shape, T(100.0));
  auto d_f = gt::empty_device<T>(f_shape);

  gt::copy(h_f, d_f);

  auto f_size = d_f.size();
  GT_DEBUG_VAR(d_f.shape());
  GT_DEBUG_VAR(f_size);

  auto d_f_noghost =
    d_f.view(gt::all, gt::all, gt::slice(nzb, -nzb), gt::slice(nvb, -nvb),
             gt::slice(nwb, -nwb), gt::all);
  auto f_noghost_size = d_f_noghost.size();
  GT_DEBUG_TYPE(d_f_noghost);
  GT_DEBUG_VAR(d_f_noghost.shape());
  GT_DEBUG_VAR(f_noghost_size);

  d_f_noghost = 1.0;

  gt::copy(d_f, h_f);

  // spot check boundary, not changed
  EXPECT_EQ(h_f(3, 3, nzb - 1, nvb, nwb, 1), T(100.0));
  EXPECT_EQ(h_f(3, 3, nzb, nvb - 1, nwb, 1), T(100.0));
  EXPECT_EQ(
    h_f(0, 1, f_shape[2] - nzb, f_shape[3] - nvb - 1, f_shape[4] - nwb - 1, 1),
    T(100.0));
  EXPECT_EQ(
    h_f(0, 1, f_shape[2] - nzb - 1, f_shape[3] - nvb, f_shape[4] - nwb - 1, 1),
    T(100.0));

  // note: interior, since nwb == 0
  EXPECT_EQ(h_f(3, 3, nzb, nvb, nwb, 1), T(1.0));
  EXPECT_EQ(
    h_f(0, 1, f_shape[2] - nzb - 1, f_shape[3] - nvb - 1, f_shape[4] - 1, 1),
    T(1.0));

  // spot check inside that was changed
  EXPECT_EQ(h_f(3, 3, nzb, nvb, nwb, 1), T(1.0));

  gt::synchronize();
}

TEST(assign, device_gfunction_mismatch_throw)
{
  using T = gt::complex<double>;

  int nzb = 2;
  int nvb = 2;
  int nwb = 0;

  // ijklmn, ghost in z, v, w
  auto f_shape = gt::shape(5, 7, 9, 11, 13, 2);

  auto g_shape =
    gt::shape(f_shape[0], f_shape[1], f_shape[2] - 2 * nzb,
              f_shape[3] - 2 * nvb, f_shape[4] - 2 * nwb, f_shape[5]);

  auto h_f = gt::full<T>(f_shape, T(100.0));
  auto d_f = gt::empty_device<T>(f_shape);
  auto h_g = gt::full(g_shape, T(2.0));
  auto d_g = gt::empty_device<T>(g_shape);

  gt::copy(h_f, d_f);
  gt::copy(h_g, d_g);

  EXPECT_THROW(h_g + h_f, std::runtime_error);
  EXPECT_THROW(d_g + d_f, std::runtime_error);
}

namespace test
{
template <typename T, gt::size_type N>
using gtensor_managed = gt::gtensor_container<gt::space::managed_vector<T>, N>;
} // end namespace test

TEST(assign, device_gene_h_from_f)
{
  using T = gt::complex<double>;

  const int nwb = 0;

  // ijklmn, ghost in w
  auto hdist_shape = gt::shape(32, 4, 48, 40, 30 + 2 * nwb, 2);
  auto fdist_shape = gt::shape(32, 4, 48, 40, 30 + 2 * nwb, 2);

  // ijklmn, no ghost
  auto prefac_shape = gt::shape(32, 4, 48, 40, 30, 2);

  // ijk + ? axis
  auto phi_shape = gt::shape(32, 4, 48, 2);

  test::gtensor_managed<T, 6> hdist(hdist_shape, T(2.0));
  test::gtensor_managed<T, 6> fdist(fdist_shape, T(1.0));
  test::gtensor_managed<T, 6> prefac(prefac_shape, T(-1.0));
  test::gtensor_managed<T, 4> phi(phi_shape, T(-2.0));
  gt::gtensor<T, 6> expected(hdist_shape, T(3.0));

  hdist.view(gt::all, gt::all, gt::all, gt::all, gt::slice(nwb, -nwb),
             gt::all) =
    fdist.view(gt::all, gt::all, gt::all, gt::all, gt::slice(nwb, -nwb),
               gt::all) +
    prefac * phi.view(gt::all, gt::all, gt::all, 0, gt::newaxis, gt::newaxis,
                      gt::newaxis);

  gt::synchronize();

  // spot check
  EXPECT_EQ(hdist, expected);
}

TEST(assign, device_broadcast_6d)
{
  gt::gtensor_device<int, 6> a(gt::shape(8, 1, 2, 4, 1, 1), 0);
  gt::gtensor_device<int, 6> b(gt::shape(8, 1, 2, 1, 1, 1), -7);

  gt::gtensor<int, 6> h_a(a.shape());

  gt::assign(a, b);

  gt::copy(a, h_a);

  for (int i = 0; i < h_a.shape(0); i++) {
    for (int j = 0; j < h_a.shape(2); j++) {
      for (int k = 0; k < h_a.shape(3); k++) {
        EXPECT_EQ(h_a(i, 0, j, k, 0, 0), -7);
      }
    }
  }
}

#endif // GTENSOR_HAVE_DEVICE
