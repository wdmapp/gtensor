
#include <benchmark/benchmark.h>

#include <gtensor/gtensor.h>

#include <gt-fft/fft.h>

#include "gt-bm.h"

using namespace gt::placeholders;

using real_t = double;
using complex_t = gt::complex<double>;

constexpr double PI = 3.141592653589793;

// ======================================================================
// BM_fft_r2c_1d
//

template <typename E, int Nx, int batch_k, typename S = gt::space::device>
static void BM_fft_r2c_1d(benchmark::State& state)
{
  int batch_size = 1024 * batch_k;
  using T = gt::complex<E>;

  auto h_A = gt::zeros<E>({Nx, batch_size});
  gt::bm::gtensor2<E, 2, S> d_A(h_A.shape());

  auto h_A2 = gt::zeros<E>(h_A.shape());
  gt::bm::gtensor2<E, 2, S> d_A2(h_A.shape());

  auto h_B = gt::empty<T>({Nx / 2 + 1, batch_size});
  auto h_B_expected = gt::empty<T>(h_B.shape());
  gt::bm::gtensor2<T, 2, S> d_B(h_B.shape());

  // Set up periodic domain with frequency 4
  // m = [sin(2*pi*x) for x in -2:1/16:2-1/16]
  double x, fx;
  for (int i = 0; i < Nx; i++) {
    x = -2.0 + i / 16.0;
    fx = sin(2 * PI * x);
    for (int j = 0; j < batch_size; j++) {
      h_A(i, j) = fx;
    }
  }

  gt::copy(h_A, d_A);

  gt::fft::FFTPlanMany<gt::fft::Domain::REAL, E> plan({Nx}, batch_size);
  std::size_t work_bytes = plan.get_work_buffer_bytes();
  std::cout << "plan work bytes: " << work_bytes << std::endl;

  auto fn = [&]() {
    plan(d_A, d_B);
    gt::synchronize();
  };

  // warm up, device compile
  fn();

  for (auto _ : state) {
    fn();
  }
}

BENCHMARK(BM_fft_r2c_1d<float, 32, 200>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_r2c_1d<double, 32, 200>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_r2c_1d<float, 64, 200>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_r2c_1d<double, 64, 200>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_r2c_1d<float, 32, 500>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_r2c_1d<double, 32, 500>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_r2c_1d<float, 64, 500>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_r2c_1d<double, 64, 500>)->Unit(benchmark::kMillisecond);

BENCHMARK(BM_fft_r2c_1d<float, 32, 200, gt::space::managed>)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_r2c_1d<double, 32, 200, gt::space::managed>)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_r2c_1d<float, 64, 200, gt::space::managed>)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_r2c_1d<double, 64, 200, gt::space::managed>)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_r2c_1d<float, 32, 500, gt::space::managed>)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_r2c_1d<double, 32, 500, gt::space::managed>)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_r2c_1d<float, 64, 500, gt::space::managed>)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_r2c_1d<double, 64, 500, gt::space::managed>)
  ->Unit(benchmark::kMillisecond);

template <typename E, int Nx, int batch_k, typename S = gt::space::device>
static void BM_fft_c2r_1d(benchmark::State& state)
{
  int batch_size = 1024 * batch_k;
  using T = gt::complex<E>;

  auto h_A = gt::zeros<E>({Nx, batch_size});
  gt::bm::gtensor2<E, 2, S> d_A(h_A.shape());

  auto h_A2 = gt::zeros<E>(h_A.shape());
  gt::bm::gtensor2<E, 2, S> d_A2(h_A.shape());

  auto h_B = gt::empty<T>({Nx / 2 + 1, batch_size});
  auto h_B_expected = gt::empty<T>(h_B.shape());
  gt::bm::gtensor2<T, 2, S> d_B(h_B.shape());

  // Set up periodic domain with frequency 4
  // m = [sin(2*pi*x) for x in -2:1/16:2-1/16]
  double x, fx;
  for (int i = 0; i < Nx; i++) {
    x = -2.0 + i / 16.0;
    fx = sin(2 * PI * x);
    for (int j = 0; j < batch_size; j++) {
      h_A(i, j) = fx;
    }
  }

  gt::copy(h_A, d_A);

  gt::fft::FFTPlanMany<gt::fft::Domain::REAL, E> plan({Nx}, batch_size);
  std::size_t work_bytes = plan.get_work_buffer_bytes();
  std::cout << "plan work bytes: " << work_bytes << std::endl;

  plan(d_A, d_B);

  auto fn = [&]() {
    plan.inverse(d_B, d_A);
    gt::synchronize();
  };

  // warm up, device compile
  fn();

  for (auto _ : state) {
    fn();
  }
}

BENCHMARK(BM_fft_c2r_1d<float, 32, 200>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_c2r_1d<double, 32, 200>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_c2r_1d<float, 64, 200>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_c2r_1d<double, 64, 200>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_c2r_1d<float, 32, 500>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_c2r_1d<double, 32, 500>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_c2r_1d<float, 64, 500>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_c2r_1d<double, 64, 500>)->Unit(benchmark::kMillisecond);

BENCHMARK(BM_fft_c2r_1d<float, 32, 200, gt::space::managed>)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_c2r_1d<double, 32, 200, gt::space::managed>)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_c2r_1d<float, 64, 200, gt::space::managed>)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_c2r_1d<double, 64, 200, gt::space::managed>)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_c2r_1d<float, 32, 500, gt::space::managed>)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_c2r_1d<double, 32, 500, gt::space::managed>)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_c2r_1d<float, 64, 500, gt::space::managed>)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_c2r_1d<double, 64, 500, gt::space::managed>)
  ->Unit(benchmark::kMillisecond);

// ======================================================================
// BM_fft_r2c_2d
//

template <typename E, int Nx, int Ny, int batch_k, typename S = gt::space::device>
static void BM_fft_r2c_2d(benchmark::State& state)
{
  int batch_size = 1024 * batch_k;
  using T = gt::complex<E>;

  auto h_A = gt::zeros<E>({Nx, Ny, batch_size});
  gt::bm::gtensor2<E, 3, S> d_A(h_A.shape());

  auto h_A2 = gt::zeros<E>(h_A.shape());
  gt::bm::gtensor2<E, 3, S> d_A2(h_A.shape());

  auto h_B = gt::empty<T>({Nx / 2 + 1, Ny, batch_size});
  auto h_B_expected = gt::empty<T>(h_B.shape());
  gt::bm::gtensor2<T, 3, S> d_B(h_B.shape());

  // Set up periodic domain with frequencies 4 and 2
  // m = [sin(2*pi*x+4*pi*y) for x in -2:1/16:2-1/16, y in 0:1/16:1-1/16]
  double x, y;
  for (int j = 0; j < Ny; j++) {
    for (int i = 0; i < Nx; i++) {
      x = -2.0 + i / 16.0;
      y = j / 16.0;
      h_A(i, j, 0) = sin(2 * PI * x + 4 * PI * y);
    }
  }

  gt::copy(h_A, d_A);

  gt::fft::FFTPlanMany<gt::fft::Domain::REAL, E> plan({Nx, Ny}, batch_size);
  std::size_t work_bytes = plan.get_work_buffer_bytes();
  std::cout << "plan work bytes: " << work_bytes << std::endl;

  auto fn = [&]() {
    plan(d_A, d_B);
    gt::synchronize();
  };

  // warm up, device compile
  fn();

  for (auto _ : state) {
    fn();
  }
}

BENCHMARK(BM_fft_r2c_2d<float, 16, 16, 200>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_r2c_2d<double, 16, 16, 200>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_r2c_2d<float, 32, 16, 200>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_r2c_2d<double, 32, 16, 200>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_r2c_2d<float, 16, 16, 500>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_r2c_2d<double, 16, 16, 500>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_r2c_2d<float, 32, 16, 500>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_r2c_2d<double, 32, 16, 500>)->Unit(benchmark::kMillisecond);

BENCHMARK(BM_fft_r2c_2d<float, 16, 16, 200, gt::space::managed>)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_r2c_2d<double, 16, 16, 200, gt::space::managed>)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_r2c_2d<float, 32, 16, 200, gt::space::managed>)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_r2c_2d<double, 32, 16, 200, gt::space::managed>)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_r2c_2d<float, 16, 16, 500, gt::space::managed>)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_r2c_2d<double, 16, 16, 500, gt::space::managed>)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_r2c_2d<float, 32, 16, 500, gt::space::managed>)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_r2c_2d<double, 32, 16, 500, gt::space::managed>)
  ->Unit(benchmark::kMillisecond);

template <typename E, int Nx, int Ny, int batch_k, typename S = gt::space::device>
static void BM_fft_c2r_2d(benchmark::State& state)
{
  int batch_size = 1024 * batch_k;
  using T = gt::complex<E>;

  auto h_A = gt::zeros<E>({Nx, Ny, batch_size});
  gt::bm::gtensor2<E, 3, S> d_A(h_A.shape());

  auto h_A2 = gt::zeros<E>(h_A.shape());
  gt::bm::gtensor2<E, 3, S> d_A2(h_A.shape());

  auto h_B = gt::empty<T>({Nx / 2 + 1, Ny, batch_size});
  auto h_B_expected = gt::empty<T>(h_B.shape());
  gt::bm::gtensor2<T, 3, S> d_B(h_B.shape());

  // Set up periodic domain with frequencies 4 and 2
  // m = [sin(2*pi*x+4*pi*y) for x in -2:1/16:2-1/16, y in 0:1/16:1-1/16]
  double x, y;
  for (int j = 0; j < Ny; j++) {
    for (int i = 0; i < Nx; i++) {
      x = -2.0 + i / 16.0;
      y = j / 16.0;
      h_A(i, j, 0) = sin(2 * PI * x + 4 * PI * y);
    }
  }

  gt::copy(h_A, d_A);

  gt::fft::FFTPlanMany<gt::fft::Domain::REAL, E> plan({Nx, Ny}, batch_size);
  std::size_t work_bytes = plan.get_work_buffer_bytes();
  std::cout << "plan work bytes: " << work_bytes << std::endl;

  plan(d_A, d_B);

  auto fn = [&]() {
    plan.inverse(d_B, d_A);
    gt::synchronize();
  };

  // warm up, device compile
  fn();

  for (auto _ : state) {
    fn();
  }
}

BENCHMARK(BM_fft_c2r_2d<float, 16, 16, 200>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_c2r_2d<double, 16, 16, 200>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_c2r_2d<float, 32, 16, 200>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_c2r_2d<double, 32, 16, 200>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_c2r_2d<float, 16, 16, 500>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_c2r_2d<double, 16, 16, 500>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_c2r_2d<float, 32, 16, 500>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_c2r_2d<double, 32, 16, 500>)->Unit(benchmark::kMillisecond);

BENCHMARK(BM_fft_c2r_2d<float, 16, 16, 200, gt::space::managed>)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_c2r_2d<double, 16, 16, 200, gt::space::managed>)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_c2r_2d<float, 32, 16, 200, gt::space::managed>)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_c2r_2d<double, 32, 16, 200, gt::space::managed>)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_c2r_2d<float, 16, 16, 500, gt::space::managed>)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_c2r_2d<double, 16, 16, 500, gt::space::managed>)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_c2r_2d<float, 32, 16, 500, gt::space::managed>)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fft_c2r_2d<double, 32, 16, 500, gt::space::managed>)
  ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
