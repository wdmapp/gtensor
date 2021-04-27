
#include <benchmark/benchmark.h>

#include <gtensor/gtensor.h>

static void BM_device_assign_4d(benchmark::State& state)
{
  auto a = gt::zeros_device<double>(gt::shape(100, 100, 100, 100));
  auto b = gt::empty_like(a);

  for (auto _ : state) {
    b = a + 2 * a;
  }
}

BENCHMARK(BM_device_assign_4d)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();