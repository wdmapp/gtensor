
#include <benchmark/benchmark.h>

#include <gtensor/gtensor.h>

using namespace gt::placeholders;

using real_t = double;
using complex_t = gt::complex<double>;

// ======================================================================
// stencil

template <int N, typename E>
inline auto stencil(E&& e, std::array<int, N> bnd, std::array<int, N> shift)
{
  std::vector<gt::gdesc> slices;
  slices.reserve(N);
  for (int d = 0; d < N; d++) {
    slices.push_back(_s(bnd[d] + shift[d], -bnd[d] + shift[d]));
  }

  return gt::view<N>(std::forward<E>(e), slices);
}

// ======================================================================
// semi_arakawa_kl_13p_v1_idep

template <typename E1, typename E2>
auto semi_arakawa_kl_13p_v1_idep(const E1& sten, const E2& a,
                                 const std::array<int, 6>& bnd)
{
  auto coeff = [&](int s) { return sten.view(_all, s, _newaxis); };

  auto rhs = coeff(0) * stencil<6>(a, bnd, {0, 0, +0, -2, 0, 0}) +
             coeff(1) * stencil<6>(a, bnd, {0, 0, -1, -1, 0, 0}) +
             coeff(2) * stencil<6>(a, bnd, {0, 0, +0, -1, 0, 0}) +
             coeff(3) * stencil<6>(a, bnd, {0, 0, +1, -1, 0, 0}) +
             coeff(4) * stencil<6>(a, bnd, {0, 0, -2, +0, 0, 0}) +
             coeff(5) * stencil<6>(a, bnd, {0, 0, -1, +0, 0, 0}) +
             coeff(6) * stencil<6>(a, bnd, {0, 0, +0, +0, 0, 0}) +
             coeff(7) * stencil<6>(a, bnd, {0, 0, +1, +0, 0, 0}) +
             coeff(8) * stencil<6>(a, bnd, {0, 0, +2, +0, 0, 0}) +
             coeff(9) * stencil<6>(a, bnd, {0, 0, -1, +1, 0, 0}) +
             coeff(10) * stencil<6>(a, bnd, {0, 0, +0, +1, 0, 0}) +
             coeff(11) * stencil<6>(a, bnd, {0, 0, +1, +1, 0, 0}) +
             coeff(12) * stencil<6>(a, bnd, {0, 0, +0, +2, 0, 0});

  return rhs;
}

// ======================================================================
// BM_semi_arakawa_kl_13p_v1_idep

static void BM_semi_arakawa_kl_13p_v1_idep(benchmark::State& state)
{
  auto shape_rhs = gt::shape(70, 32, 24, 24, 32, 2);
  auto shape_sten = gt::shape(70, 13, 24, 24, 32, 2);

  std::array<int, 6> bnd = {0, 0, 2, 2, 0, 0};
  gt::shape_type<6> shape_f;
  for (int d = 0; d < 6; d++) {
    shape_f[d] = shape_rhs[d] + 2 * bnd[d];
  }

  auto rhs = gt::zeros_device<complex_t>(shape_rhs);
  auto f = gt::zeros_device<complex_t>(shape_f);
  auto sten = gt::zeros_device<real_t>(shape_sten);

  for (auto _ : state) {
    rhs = rhs + semi_arakawa_kl_13p_v1_idep(sten, f, bnd);
    gt::synchronize();
  }
}

BENCHMARK(BM_semi_arakawa_kl_13p_v1_idep)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
