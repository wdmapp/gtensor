
#include <benchmark/benchmark.h>

#include <gtensor/gtensor.h>

using namespace gt::placeholders;

using real_t = double;
using complex_t = gt::complex<double>;

// ======================================================================
// BM_device_assign_4d

static void BM_device_assign_4d(benchmark::State& state)
{
  auto a = gt::zeros_device<real_t>(gt::shape(10, 10, 10, 10));
  auto b = gt::empty_like(a);

  for (auto _ : state) {
    b = a + 2 * a;
    gt::synchronize();
  }
}

BENCHMARK(BM_device_assign_4d)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();