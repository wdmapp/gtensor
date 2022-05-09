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

//#define READ_INPUT

template <typename T>
inline void read_carray(std::ifstream& f, int n, T* Adata)
{
  for (int i = 0; i < n; i++) {
    // std::cout << i << " " << std::endl;
    f >> Adata[i];
  }
}

inline void read_iarray(std::ifstream& f, int n, gt::blas::index_t* data)
{
  for (int i = 0; i < n; i++) {
    f >> data[i];
  }
}

#define MAX_SIZE_STR_LEN 15

template <typename T>
void test(int n, int nrhs, int batch_size, int bw)
{
  int lda, ldb;

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

  using CT = gt::complex<T>;

#ifdef READ_INPUT
  std::ifstream f("zgetrs.txt", std::ifstream::in);

  f >> n;
  f >> nrhs;
  f >> lda;
  f >> ldb;
  f >> batch_size;

#else
  lda = n;
  ldb = n;
#endif

  ss << n << "x" << n << "x" << nrhs << "x" << batch_size;
  size_str = ss.str();

#if 0
  std::cout << "n    = " << n << std::endl;
  std::cout << "nrhs = " << nrhs << std::endl;
  std::cout << "lda  = " << lda << std::endl;
  std::cout << "ldb  = " << ldb << std::endl;
  std::cout << "batch_size = " << batch_size << std::endl;
#endif

  auto h_Aptr = gt::empty<CT*>({batch_size});
  auto h_Ainvptr = gt::empty<CT*>({batch_size});
  auto h_Bptr = gt::empty<CT*>({batch_size});
  auto h_Cptr = gt::empty<CT*>({batch_size});
  auto d_Aptr = gt::empty_device<CT*>({batch_size});
  auto d_Ainvptr = gt::empty_device<CT*>({batch_size});
  auto d_Bptr = gt::empty_device<CT*>({batch_size});
  auto d_Cptr = gt::empty_device<CT*>({batch_size});

  auto h_Adata = gt::zeros<CT>({n, n, batch_size});
  auto h_Ainvdata = gt::zeros<CT>({n, n, batch_size});
  auto h_Bdata = gt::zeros<CT>({n, nrhs, batch_size});
  auto h_Cdata = gt::zeros<CT>({n, nrhs, batch_size});
  auto d_Adata = gt::empty_device<CT>(h_Adata.shape());
  auto d_Ainvdata = gt::empty_device<CT>(h_Adata.shape());
  auto d_Bdata = gt::empty_device<CT>(h_Bdata.shape());
  auto d_Cdata = gt::empty_device<CT>(h_Cdata.shape());

  auto h_piv = gt::empty<gt::blas::index_t>({n, batch_size});
  auto d_piv = gt::empty_device<gt::blas::index_t>(h_piv.shape());

  auto info = gt::zeros<int>({batch_size});

  std::cout << "INFO: allocate done" << std::endl;

#ifdef READ_INPUT
  read_carray(f, n * n * batch_size, h_Adata.data());
  read_carray(f, n * nrhs * batch_size, h_Bdata.data());
  read_iarray(f, n * batch_size, h_piv.data());
  f.close();
  std::cout << "INFO: read done" << std::endl;
#else
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
  std::cout << "INFO: init done" << std::endl;
#endif

  for (int i = 0; i < batch_size; i++) {
    h_Aptr(i) = gt::raw_pointer_cast(d_Adata.data()) + (n * n * i);
    h_Ainvptr(i) = gt::raw_pointer_cast(d_Ainvdata.data()) + (n * n * i);
    h_Bptr(i) = gt::raw_pointer_cast(d_Bdata.data()) + (n * nrhs * i);
    h_Cptr(i) = gt::raw_pointer_cast(d_Cdata.data()) + (n * nrhs * i);
  }
  gt::copy(h_Aptr, d_Aptr);
  gt::copy(h_Adata, d_Adata);
  gt::copy(h_Ainvptr, d_Ainvptr);
  gt::copy(h_Bptr, d_Bptr);
  gt::copy(h_Bdata, d_Bdata);
  gt::copy(h_Cptr, d_Cptr);
  gt::copy(h_piv, d_piv);

  std::cout << "INFO: memcpy to device done" << std::endl;

  gt::blas::handle_t* h = gt::blas::create();

  struct timespec start, end;
  double elapsed, total = 0.0;
  int info_sum;

  auto bw2 = gt::blas::get_max_bandwidth(n, gt::raw_pointer_cast(d_Aptr.data()),
                                         lda, batch_size);
#ifndef READ_INPUT
  if (bw2.lower != bw || bw2.upper != bw) {
    std::cerr << "ERR: construct matrix bandwidht mismatch, expected " << bw
              << " got " << bw2.lower << "_" << bw2.upper << std::endl;
    std::exit(1);
  }
#endif
  ss.str("");
  ss << bw2.lower << "_" << bw2.upper;
  bw_str = ss.str();

  gt::blas::invert_banded_batched(n, gt::raw_pointer_cast(d_Aptr.data()), lda,
                                  gt::raw_pointer_cast(d_piv.data()),
                                  gt::raw_pointer_cast(d_Ainvptr.data()), lda,
                                  batch_size, bw2.lower, bw2.upper);

  for (int i = 0; i < NRUNS; i++) {
    clock_gettime(CLOCK_MONOTONIC, &start);
    gt::blas::getrs_batched(h, n, nrhs, gt::raw_pointer_cast(d_Aptr.data()),
                            lda, gt::raw_pointer_cast(d_piv.data()),
                            gt::raw_pointer_cast(d_Bptr.data()), ldb,
                            batch_size);
    gt::synchronize();
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed =
      (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1.0e-9;
    if (i > 0)
      total += elapsed;
    info_sum = gt::sum(info);
    if (info_sum != 0)
      std::cout << "INFO: info sum: " << info_sum << std::endl;
    std::cout << "INFO: run blas [" << i << "]: " << elapsed << std::endl;
  }
  std::cout << "type\tsize\tlbw_ubw\talgorithm\tseconds" << std::endl;
  std::cout << type_str << "\t" << size_str << "\t" << bw_str
            << "\tzgetrs_avg\t" << total / (NRUNS - 1) << std::endl;

  total = 0.0;
  for (int i = 0; i < NRUNS; i++) {
    clock_gettime(CLOCK_MONOTONIC, &start);
    gt::blas::getrs_banded_batched(n, nrhs, gt::raw_pointer_cast(d_Aptr.data()),
                                   lda, gt::raw_pointer_cast(d_piv.data()),
                                   gt::raw_pointer_cast(d_Bptr.data()), ldb,
                                   batch_size, bw2.lower, bw2.upper);
    gt::synchronize();
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed =
      (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1.0e-9;
    if (i > 0)
      total += elapsed;
    std::cout << "INFO: run band [" << i << "]: " << elapsed << std::endl;
  }

  std::cout << type_str << "\t" << size_str << "\t" << bw_str
            << "\tbanded_avg\t" << total / (NRUNS - 1) << std::endl;

  total = 0.0;
  for (int i = 0; i < NRUNS; i++) {
    clock_gettime(CLOCK_MONOTONIC, &start);
    gt::blas::gemm_batched<CT>(
      h, n, nrhs, n, 1.0, gt::raw_pointer_cast(d_Ainvptr.data()), lda,
      gt::raw_pointer_cast(d_Bptr.data()), ldb, 0.0,
      gt::raw_pointer_cast(d_Cptr.data()), ldb, batch_size);
    gt::synchronize();
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed =
      (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1.0e-9;
    if (i > 0)
      total += elapsed;
    std::cout << "INFO: run invt [" << i << "]: " << elapsed << std::endl;
  }

  std::cout << type_str << "\t" << size_str << "\t" << bw_str
            << "\tinverted_avg\t" << total / (NRUNS - 1) << std::endl;

// needs update for change to use non-trivial input with specified bw
#if 0
  // check result
  gt::copy(d_Bdata, h_Bdata);
  bool ok = true;
  CT err = CT(0.0, 0.0);
  for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < nrhs; j++) {
        err = h_Bdata(i, j, b) - CT(i / (j + 1) * b, i * j / (b + 1));
        if (gt::abs(err) > 0.0) {
          std::cout << "ERR: err of " << err << " at [" << b << ", " << i
                    << ", " << j << "]" << std::endl;
          ok = false;
          break;
        }
      }
      if (!ok) {
        break;
      }
    }
    if (!ok) {
      break;
    }
  }
#endif

  gt::blas::destroy(h);

  std::cout << "INFO: destroy done" << std::endl;
}

int main(int argc, char** argv)
{
#ifdef GTENSOR_DEVICE_HIP
  rocblas_initialize();
#endif

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

  // size used for single GPU GENE run
  test<float>(n, nrhs, batch_size, bw);
  test<double>(n, nrhs, batch_size, bw);
}
