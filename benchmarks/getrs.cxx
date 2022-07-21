#include <complex>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <time.h>

#include "gt-blas/blas.h"
#include "gtensor/gtensor.h"
#include "gtensor/reductions.h"

#define NRUNS 10

template <typename CT>
struct test_problem
{
  gt::gtensor<CT, 3> h_Adata;
  int lda;
  gt::gtensor<CT, 3> h_Bdata;
  int ldb;
  gt::gtensor<gt::blas::index_t, 2> h_piv;
};

template <typename T>
inline void read_array(std::ifstream& f, int n, T* data)
{
  for (int i = 0; i < n; i++) {
    f >> data[i];
  }
}

template <typename F>
double bench(int nruns, F f, char const* label = "")
{
  timespec start, end;
  double elapsed, total = 0.0;
  for (int i = 0; i < nruns; i++) {
    clock_gettime(CLOCK_MONOTONIC, &start);
    f();
    gt::synchronize();
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed =
      (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1.0e-9;
    total += elapsed;
    std::cout << "INFO: run " << label << " [" << i << "]: " << elapsed
              << std::endl;
  }
  return total / nruns;
}

/**
 * @brief Check residual
 *
 * We are given P L U x = b, where P, L, U, and b are part of test_problem
 * and x is given by d_Bdata.
 * The function checks whether ||P L U x - b||_1 / ||b||_1 < tol.
 */
template <typename T, typename CT = gt::complex<T>>
bool check(test_problem<CT> const& tp,
           gt::gtensor<CT, 3, gt::space::device> const& d_Bdata,
           char const* label = "")
{
  constexpr T tol = 100.0 * std::numeric_limits<T>::epsilon();
  auto h_sol = gt::zeros<CT>(tp.h_Bdata.shape());
  gt::copy(d_Bdata, h_sol);
  std::size_t n = tp.h_Adata.shape(0);
  std::size_t batch_size = tp.h_Adata.shape(2);
  std::size_t nrhs = tp.h_Bdata.shape(1);
  auto tmp = gt::zeros<CT>({static_cast<int>(n)});
  auto rhs = gt::zeros<CT>({static_cast<int>(n)});
  for (gt::blas::index_t r = 0; r < nrhs; ++r) {
    for (gt::blas::index_t b = 0; b < batch_size; ++b) {
      T residual_L1 = 0.0;
      T b_L1 = 0.0;
      rhs = tp.h_Bdata.view(gt::all, r, b);
      // swap rows in right-hand side, computes rhs = P^T b
      for (gt::blas::index_t i = 0; i < n; ++i) {
        auto ip = tp.h_piv(i, b) - 1;
        if (i != ip) {
          std::swap(rhs(i), rhs(ip));
        }
      }
      // Compute tmp = U x
      for (gt::blas::index_t i = 0; i < n; ++i) {
        tmp(i) = CT{};
        for (gt::blas::index_t j = i; j < n; ++j) {
          tmp(i) += tp.h_Adata(i, j, b) * h_sol(j, r, b);
        }
      }
      // Compute ||L tmp - rhs||_1 and ||rhs||_1
      for (gt::blas::index_t i = 0; i < n; ++i) {
        auto b_i = tmp(i);
        for (gt::blas::index_t j = 0; j < i; ++j) {
          b_i += tp.h_Adata(i, j, b) * tmp(j);
        }
        residual_L1 += gt::abs(b_i - rhs(i));
        b_L1 += gt::abs(rhs(i));
      }
      // check residual
      if (residual_L1 / b_L1 > tol) {
        std::cerr << label << " relative error @ (" << r << "," << b
                  << "): " << residual_L1 / b_L1 << " > " << tol << " ("
                  << residual_L1 << "/" << b_L1 << ")" << std::endl;
        return false;
      }
    }
  }
  return true;
}

template <typename T, typename CT = gt::complex<T>>
auto make_test_problem(int n, int nrhs, int batch_size, int bw)
  -> test_problem<CT>
{
  auto h_Adata = gt::zeros<CT>({n, n, batch_size});
  auto h_Bdata = gt::zeros<CT>({n, nrhs, batch_size});
  auto h_piv = gt::empty<gt::blas::index_t>({n, batch_size});
  for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < n; i++) {
      h_Adata(i, i, b) = CT(bw + 1.0, 0.0);
      h_piv(i, b) = i + 1;
      // set upper / lower diags at bw diagonal
      if (i + bw < n) {
        h_Adata(i, i + bw, b) = CT(-1.0, 0.0);
        h_Adata(i + bw, i, b) = CT(0.0, -1.0);
      }
      for (int j = 0; j < nrhs; j++) {
        // h_Bdata(i, j, b) = CT(i / (j + 1) * b, i * j / (b + 1));
        h_Bdata(i, j, b) = CT(1.0, 0.0);
      }
    }
  }
  return {h_Adata, n, h_Bdata, n, h_piv};
}

template <typename T, typename CT = gt::complex<T>>
auto read_test_problem(char const* file_name) -> test_problem<CT>
{
  auto f = std::ifstream(file_name, std::ifstream::in);
  if (!f.is_open()) {
    std::cerr << "Error: Could not open " << file_name << std::endl;
    return {};
  }

  int n, nrhs, lda, ldb, batch_size;
  f >> n;
  f >> nrhs;
  f >> lda;
  f >> ldb;
  f >> batch_size;

  auto h_Adata = gt::zeros<CT>({n, n, batch_size});
  auto h_Bdata = gt::zeros<CT>({n, nrhs, batch_size});
  auto h_piv = gt::empty<gt::blas::index_t>({n, batch_size});
  read_array(f, n * n * batch_size, h_Adata.data());
  read_array(f, n * nrhs * batch_size, h_Bdata.data());
  read_array(f, n * batch_size, h_piv.data());
  f.close();

  return {h_Adata, lda, h_Bdata, ldb, h_piv};
}

template <typename T, typename CT = gt::complex<T>>
void test(test_problem<CT> tp, int known_bw = 0)
{
  int n = tp.h_Adata.shape(0);
  int nrhs = tp.h_Bdata.shape(1);
  int batch_size = tp.h_Adata.shape(2);
  int lda = tp.lda;
  int ldb = tp.ldb;

  std::ostringstream ss;
  std::string size_str;
  std::string type_str;
  std::string bw_str;

  std::cout.precision(10);

  if (std::is_same<T, float>::value) {
    type_str = "float ";
  } else if (std::is_same<T, double>::value) {
    type_str = "double";
  } else {
    type_str = "unkown";
  }

  ss << n << "x" << n << "x" << nrhs << "x" << batch_size;
  size_str = ss.str();

  auto h_Aptr = gt::empty<CT*>({batch_size});
  auto h_Ainvptr = gt::empty<CT*>({batch_size});
  auto h_Bptr = gt::empty<CT*>({batch_size});
  auto h_Cptr = gt::empty<CT*>({batch_size});
  auto d_Aptr = gt::empty_device<CT*>({batch_size});
  auto d_Ainvptr = gt::empty_device<CT*>({batch_size});
  auto d_Bptr = gt::empty_device<CT*>({batch_size});
  auto d_Cptr = gt::empty_device<CT*>({batch_size});

  auto h_Ainvdata = gt::zeros<CT>({n, n, batch_size});
  auto h_Cdata = gt::zeros<CT>({n, nrhs, batch_size});
  auto d_Adata = gt::empty_device<CT>(tp.h_Adata.shape());
  auto d_Ainvdata = gt::empty_device<CT>(tp.h_Adata.shape());
  auto d_Bdata = gt::empty_device<CT>(tp.h_Bdata.shape());
  auto d_Cdata = gt::empty_device<CT>(h_Cdata.shape());

  auto d_piv = gt::empty_device<gt::blas::index_t>(tp.h_piv.shape());

  std::cout << "INFO: allocate done" << std::endl;

  for (int i = 0; i < batch_size; i++) {
    h_Aptr(i) = gt::raw_pointer_cast(d_Adata.data()) + (n * n * i);
    h_Ainvptr(i) = gt::raw_pointer_cast(d_Ainvdata.data()) + (n * n * i);
    h_Bptr(i) = gt::raw_pointer_cast(d_Bdata.data()) + (n * nrhs * i);
    h_Cptr(i) = gt::raw_pointer_cast(d_Cdata.data()) + (n * nrhs * i);
  }
  gt::copy(h_Aptr, d_Aptr);
  gt::copy(tp.h_Adata, d_Adata);
  gt::copy(h_Ainvptr, d_Ainvptr);
  gt::copy(h_Bptr, d_Bptr);
  gt::copy(h_Cptr, d_Cptr);
  gt::copy(tp.h_piv, d_piv);

  std::cout << "INFO: memcpy to device done" << std::endl;

  gt::blas::handle_t* h = gt::blas::create();

  auto bw2 = gt::blas::get_max_bandwidth(
    h, n, gt::raw_pointer_cast(d_Aptr.data()), lda, batch_size);
  if (known_bw > 0) {
    if (bw2.lower != known_bw || bw2.upper != known_bw) {
      std::cerr << "ERR: construct matrix bandwidth mismatch, expected "
                << known_bw << " got " << bw2.lower << "_" << bw2.upper
                << std::endl;
      std::exit(1);
    }
  }
  ss.str("");
  ss << bw2.lower << "_" << bw2.upper;
  bw_str = ss.str();

  gt::blas::invert_banded_batched(h, n, gt::raw_pointer_cast(d_Aptr.data()),
                                  lda, gt::raw_pointer_cast(d_piv.data()),
                                  gt::raw_pointer_cast(d_Ainvptr.data()), lda,
                                  batch_size, bw2.lower, bw2.upper);

  auto const test_blas = [&]() {
    gt::blas::getrs_batched(h, n, nrhs, gt::raw_pointer_cast(d_Aptr.data()),
                            lda, gt::raw_pointer_cast(d_piv.data()),
                            gt::raw_pointer_cast(d_Bptr.data()), ldb,
                            batch_size);
  };
  auto const test_banded = [&]() {
    gt::blas::getrs_banded_batched(
      h, n, nrhs, gt::raw_pointer_cast(d_Aptr.data()), lda,
      gt::raw_pointer_cast(d_piv.data()), gt::raw_pointer_cast(d_Bptr.data()),
      ldb, batch_size, bw2.lower, bw2.upper);
  };
  auto const test_inverted = [&]() {
    gt::blas::gemm_batched<CT>(
      h, n, nrhs, n, 1.0, gt::raw_pointer_cast(d_Ainvptr.data()), lda,
      gt::raw_pointer_cast(d_Bptr.data()), ldb, 0.0,
      gt::raw_pointer_cast(d_Cptr.data()), ldb, batch_size);
    copy(d_Cdata, d_Bdata);
  };

  auto check_and_measure = [&](auto test_fun, char const* name) {
    gt::copy(tp.h_Bdata, d_Bdata);
    test_fun();
    gt::synchronize();
    if (check<T>(tp, d_Bdata, name)) {
      double time = bench(NRUNS, test_fun, name);
      std::cout << type_str << "\t" << size_str << "\t" << bw_str << "\t"
                << name << "_avg\t" << time << std::endl;
    }
  };

  std::cout << "type\tsize\tlbw_ubw\talgorithm\tseconds" << std::endl;
  check_and_measure(test_blas, "blas");
  check_and_measure(test_banded, "banded");
  check_and_measure(test_inverted, "inverted");

  gt::blas::destroy(h);

  std::cout << "INFO: destroy done" << std::endl;
}

int main(int argc, char** argv)
{
#ifdef GTENSOR_DEVICE_HIP
  rocblas_initialize();
#endif

  if (argc >= 2 && strcmp(argv[1], "file") == 0) {
    if (argc >= 3) {
      test<float>(read_test_problem<float>(argv[2]));
      test<double>(read_test_problem<double>(argv[2]));
      return 0;
    }
    return -1;
  } else {
    int n = 140;
    int nrhs = 1;
    int batch_size = 384;
    int bw = 32;

    if (argc > 1) {
      n = std::stoi(argv[1]);
    }
    if (argc > 2) {
      nrhs = std::stoi(argv[2]);
    }
    if (argc > 3) {
      batch_size = std::stoi(argv[3]);
    }
    if (argc > 4) {
      bw = std::stoi(argv[4]);
    }
    //// size used for single GPU GENE run
    test<float>(make_test_problem<float>(n, nrhs, batch_size, bw), bw);
    test<double>(make_test_problem<double>(n, nrhs, batch_size, bw), bw);
  }
  return 0;
}
