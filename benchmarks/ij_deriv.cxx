#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <type_traits>

#include <gtensor/gtensor.h>

using namespace gt::placeholders;

template <typename T>
using host_vector = gt::backend::host_storage<T>;
template <typename T>
using device_vector = gt::backend::device_storage<T>;

template <typename Real>
void compare_deriv(Real* error, Real* maxError, Real* relError,
                   Real* maxRelError, gt::complex<Real>* a,
                   gt::complex<Real>* b, int imin, int imax, int jmin, int jmax,
                   int kmin, int kmax, int deriv, int ilen, int jlen, int klen,
                   int nderiv)
{
  using Complex = gt::complex<Real>;

  int idx, nelements = 0;
  Complex a1, b1;

  // error = sqrt(sum((sol-df)**2)/size
  // maxError = maxval(abs(sol-df))
  *error = 0;
  *maxError = 0;
  *relError = 0;
  *maxRelError = 0;
  double diffNorm, rel;
  for (int i = imin; i < imax; i++) {
    for (int j = jmin; j < jmax; j++) {
      for (int k = kmin; k < kmax; k++) {
        idx = k * nderiv * jlen * ilen + deriv * jlen * ilen + j * ilen + i;
        nelements++;
        a1 = a[idx];
        b1 = b[idx];
        diffNorm = abs(a1 - b1);
        // calculate |(b1 - a1) / a1| using |b1/a1 - 1|
        if (a1.real() == 0 && a1.imag() == 0) {
          rel = abs(b1);
        } else {
          rel = abs((b1 / a1) - Complex(1.0, 0.0));
        }
        *error += diffNorm;
        *relError += rel;
        if (diffNorm > *maxError)
          *maxError = diffNorm;
        if (rel > *maxRelError)
          *maxRelError = rel;
        if (0 && diffNorm > (16 * DBL_EPSILON) && rel > 1e-7) {
          printf("Big error: [%d,%d,%d,%d], (%0.4e, %0.4e); "
                 "(%.04e, %0.4e) %0.4e\n",
                 i, j, deriv, k, a1.real(), a1.imag(), b1.real(), b1.imag(),
                 rel);
        }
      }
    }
  }
  *error = *error / nelements;
  *relError = *relError / nelements;
}

template <typename Real>
void compare_cmplx(Real* error, Real* maxError, Real* relError,
                   Real* maxRelError, gt::complex<Real>* a,
                   gt::complex<Real>* b, int size)
{
  using Complex = gt::complex<Real>;

  // error = sqrt(sum((sol-df)**2)/size
  // maxError = maxval(abs(sol-df))
  *error = 0;
  *maxError = 0;
  *relError = 0;
  *maxRelError = 0;
  Real diffNorm, rel;
  for (int i = 0; i < size; i++) {
    diffNorm = abs(a[i] - b[i]);
    // calculate |(b[i] - a[i]) / a[i]| using |b[i]/a[i] - 1|
    if (a[i].real() == 0 && a[i].imag() == 0) {
      rel = abs(b[i]);
    } else {
      rel = abs((b[i] / a[i]) - Complex(1.0, 0.0));
    }
    *error += diffNorm;
    *relError += rel;
    if (diffNorm > *maxError)
      *maxError = diffNorm;
    if (rel > *maxRelError)
      *maxRelError = rel;
    if (diffNorm > (2 * DBL_EPSILON) && rel > 1e-7) {
      printf("Big error: %d, (%0.2e, %0.2e); (%.02e, %0.2e) %0.2e\n", i,
             a[i].real(), a[i].imag(), b[i].real(), b[i].imag(), rel);
    }
  }
  *error = *error / size;
  *relError = *relError / size;
}

template <typename Real>
void array_stats(Real* minNorm, Real* maxNorm, Real* minRe, Real* maxRe,
                 Real* minIm, Real* maxIm, gt::complex<Real>* a, int size)
{
  Real norm, x, y;
  Real minVal, maxVal;

  if (std::is_same<Real, double>::value) {
    minVal = DBL_MIN;
    maxVal = DBL_MAX;
  } else {
    minVal = FLT_MIN;
    maxVal = FLT_MAX;
  }

  *minNorm = maxVal;
  *maxNorm = minVal;
  *minRe = maxVal;
  *maxRe = minVal;
  *minIm = maxVal;
  *maxIm = minVal;
  for (int i = 0; i < size; i++) {
    x = a[i].real();
    y = a[i].imag();

    if (x > *maxRe)
      *maxRe = x;
    if (x < *minRe && x != 0)
      *minRe = x;

    if (y > *maxIm)
      *maxIm = y;
    if (y < *minIm && y != 0)
      *minIm = y;

    norm = sqrt(pow(x, 2) + pow(y, 2));
    if (norm > *maxNorm)
      *maxNorm = norm;
    if (norm < *minNorm && norm != 0)
      *minNorm = norm;
  }

  if (*minRe == maxVal)
    *minRe = 0.0;
  if (*maxRe == minVal)
    *maxRe = 0.0;
  if (*minIm == maxVal)
    *minIm = 0.0;
  if (*maxIm == minVal)
    *maxIm = 0.0;
}

/* Compute the i and j derivative of arr and write it to darr(:,:,1:2,:)
   arr has nb boundary points in the first dimension which has to match the
   number of coefficient: nb=(ncoeff-1)/2 INPUT: arr(dim1+2*nb,dim2,dim3)
          coeff(ncoeff), the coefficients for the stencil in i direction
          ikj(dim2), i*k_j for computation of the j-derivative
   OUTPUT: darr(dim1,dim2,1:2,dim3), where the third dimension contains the
           i-derivative (1) and the j-derivative (2)
 */
template <typename Real>
void ij_deriv_cpu(const int dim1, const int dim2, const int dim3,
                  const gt::complex<Real>* arr, const int ncoeff,
                  const Real* coeff, const gt::complex<Real>* ikj,
                  gt::complex<Real>* darr)
{
  using Complex = gt::complex<Real>;

  Complex tmp;

  // launch one thread block per ij slice as they are independent
  int klmn, i, j, sten, wbdim1 = dim1 + ncoeff - 1;
  int nb = (ncoeff - 1) / 2;

#define IDX3(a, b, c) (c * dim2 * wbdim1 + b * wbdim1 + a)
#define IDX4(a, b, z, c) (c * 2 * dim2 * dim1 + z * dim2 * dim1 + b * dim1 + a)

  for (klmn = 0; klmn < dim3; klmn++) {
    for (j = 0; j < dim2; j++) {
      for (i = 0; i < dim1; i++) {
        tmp = Complex(0.0, 0.0);
        for (sten = 0; sten < ncoeff; sten++) {
          tmp = tmp + coeff[sten] * arr[IDX3(i + sten, j, klmn)];
        }
        darr[IDX4(i, j, 0, klmn)] = tmp;
        darr[IDX4(i, j, 1, klmn)] = ikj[j] * arr[IDX3(i + nb, j, klmn)];
      }
    }
  }
}

template <typename Real, typename Space>
void ij_deriv_gt(const gt::gtensor_span<const gt::complex<Real>, 3, Space>& arr,
                 const gt::gtensor_span<const Real, 1, gt::space::host>& coeff,
                 const gt::gtensor_span<const gt::complex<Real>, 1, Space>& ikj,
                 gt::gtensor_span<gt::complex<Real>, 4, Space>& darr)
{
  assert(coeff.shape(0) == 5);
  assert(arr.shape(0) - darr.shape(0) == 4);

  int nb = (coeff.shape(0) - 1) / 2;
  int li0 = darr.shape(0);

  // x-derivative
  darr.view(_all, _all, 0, _all) =
    coeff(0) * arr.view(_s(0, li0 + 0), _all, _all) +
    coeff(1) * arr.view(_s(1, li0 + 1), _all, _all) +
    coeff(2) * arr.view(_s(2, li0 + 2), _all, _all) +
    coeff(3) * arr.view(_s(3, li0 + 3), _all, _all) +
    coeff(4) * arr.view(_s(4, li0 + 4), _all, _all);

  // y-derivative
  darr.view(_all, _all, 1, _all) =
    ikj.view(_newaxis, _all, _newaxis) * arr.view(_s(nb, -nb), _all, _all);
}

/* Compute the i and j derivative of arr and write it to darr(:,:,1:2,:)
   arr has nb boundary points in the first dimension which has to match the
   number of coefficients: nb=(ncoeff-1)/2
   INPUT: arr(dim1+2*nb,dim2,dim3)
          coeff(ncoeff), the real coefficients for the stencil in i direction
              ikj(dim2), i*k_j complex coefficients for the j-derivative
   OUTPUT: darr(dim1,dim2,1:2,dim3), where the third dimension contains the
           i-derivative (1) and the j-derivative (2)
 */
template <typename Real>
void ij_deriv_gt_device(const int dim1, const int dim2, const int dim3,
                        const gt::complex<Real>* arr_, const int ncoeff,
                        const Real* coeff_, const gt::complex<Real>* ikj_,
                        gt::complex<Real>* darr_)
{
  int nb = (ncoeff - 1) / 2;

  // Note: use host array for coeffecients, so they will be copied in to
  // the kernel lambda as constants
  auto coeff = gt::adapt<1>(coeff_, gt::shape(ncoeff));

  auto arr = gt::adapt_device<3>(arr_, gt::shape(dim1 + 2 * nb, dim2, dim3));
  auto ikj = gt::adapt_device<1>(ikj_, gt::shape(dim2));
  auto darr = gt::adapt_device<4>(darr_, gt::shape(dim1, dim2, 2, dim3));

  ij_deriv_gt(arr, coeff, ikj, darr);
  gt::backend::device_synchronize();
}

template <typename Real>
void ij_deriv_gt_host(const int dim1, const int dim2, const int dim3,
                      const gt::complex<Real>* arr_, const int ncoeff,
                      const Real* coeff_, const gt::complex<Real>* ikj_,
                      gt::complex<Real>* darr_)
{
  int nb = (ncoeff - 1) / 2;

  // Note: use host array for coeffecients, so they will be copied in to
  // the kernel lambda as constants
  auto coeff = gt::adapt<1>(coeff_, gt::shape(ncoeff));

  auto arr = gt::adapt<3>(arr_, gt::shape(dim1 + 2 * nb, dim2, dim3));
  auto ikj = gt::adapt<1>(ikj_, gt::shape(dim2));
  auto darr = gt::adapt<4>(darr_, gt::shape(dim1, dim2, 2, dim3));

  ij_deriv_gt(arr, coeff, ikj, darr);
}

#if defined(GTENSOR_DEVICE_CUDA) || defined(GTENSOR_DEVICE_HIP)
template <typename Real>
__global__ void ij_deriv_kernel(const int len1, const int len2, const int len3,
                                const gt::complex<Real>* arr, const int ncoeff,
                                const Real* coeff, const gt::complex<Real>* ikj,
                                gt::complex<Real>* darr)
{
  int i, j, k, sten;
  int idx, didx1, didx2;
  int wblen1 = len1 + ncoeff - 1;
  int nb = (ncoeff - 1) / 2;
  gt::complex<Real> tmp;

  k = blockIdx.z;
  j = blockIdx.y;
  i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < len1 && j < len2 && k < len3) {
    idx = k * len2 * wblen1 + j * wblen1 + i;
    didx1 = k * 2 * len2 * len1 + j * len1 + i;
    didx2 = didx1 + len2 * len1;

    tmp = coeff[0] * arr[idx];
    for (sten = 1; sten < ncoeff; sten++) {
      tmp = tmp + (coeff[sten] * arr[idx + sten]);
    }
    darr[didx1] = tmp;
    darr[didx2] = ikj[j] * arr[idx + nb];
  }
}

template <typename Real>
void ij_deriv_gpu(const int len1, // 16-1024
                  const int len2, // 1-256
                  const int len3, // 2-64
                  const gt::complex<Real>* arr, const int ncoeff,
                  const Real* coeff, const gt::complex<Real>* ikj,
                  gt::complex<Real>* darr)
{
  // todo: calculate block dim based on dim1 and warp size - pick
  // reasonable multiple of 32 that is close to multiple of dim1,
  // less then 1024
  dim3 dimBlock(256, 1, 1);
  dim3 dimGrid(ceil(len1 / 256.0), len2, len3);

  gtLaunchKernel(ij_deriv_kernel, dimGrid, dimBlock, 0, 0, len1, len2, len3,
                 arr, ncoeff, coeff, ikj, darr);
  gt::backend::device_synchronize();
}
#endif // end CUDA or HIP

template <typename Real>
void test_ij_deriv(int li0, int lj0, int lbg0)
{
  using Complex = gt::complex<Real>;

  constexpr int time_warmup_count = 20;
  constexpr int time_run_count = 20;
  struct timespec start, end;
  double seconds_per_run = 0.0;

  int ncoeff, nxb, lx0, i, j, klmn;
  Real pi = 3.14159265358979323846;

#ifdef DEBUG_COMPARE
  Real error, maxError, relError, maxRelError;
  Real minNorm, maxNorm, minRe, maxRe, minIm, maxIm;
#endif

  ncoeff = 5;
  nxb = ncoeff / 2;
  lx0 = li0 + (2 * nxb);

  size_t arr_size = lx0 * lj0 * lbg0;
  size_t darr_size = li0 * lj0 * 2 * lbg0;
  size_t ikj_size = lj0;

  printf("== %dx%dx%d ==\n", li0, lj0, lbg0);

  host_vector<Complex> h_arr(arr_size);
  device_vector<Complex> d_arr(arr_size);

  host_vector<Complex> h_darr(darr_size);
  device_vector<Complex> d_darr(darr_size);

  host_vector<Complex> ref_darr(darr_size);

  host_vector<Complex> h_ikj(ikj_size);
  device_vector<Complex> d_ikj(ikj_size);

  host_vector<Real> h_coeff(ncoeff);
  device_vector<Real> d_coeff(ncoeff);

  // initialize the input arrays
  // 4th order centered difference
  h_coeff[0] = 1.0 / 12.0;
  h_coeff[1] = -2.0 / 3.0;
  h_coeff[2] = 0.0;
  h_coeff[3] = 2.0 / 3.0;
  h_coeff[4] = -1.0 / 12.0;

  gt::backend::copy(h_coeff, d_coeff);

#define ARRIDX(a, b, c) (c * lj0 * lx0 + b * lx0 + a)

  // Initialize input array on host
  for (klmn = 0; klmn < lbg0; klmn++) {
    for (j = 0; j < lj0; j++) {
      for (i = 0; i < lx0; i++) {
        if (klmn < 3) {
          h_arr[ARRIDX(i, j, klmn)] =
            (pow(0.1 * (i - nxb) - 0.5, (klmn + 1)) + 0.3 * (j + 1),
             pow(0.2 * (i - nxb) - 0.5, (klmn + 1)) + 0.6 * (j + 1));
        } else {
          h_arr[ARRIDX(i, j, klmn)] =
            (5.0 * j * sin((klmn + 1) * i / li0 * 2.0 * pi),
             2.5 * j * sin((klmn + 1) * i / li0 * 1.2 * pi));
        }
      }
    }
  }

  gt::backend::copy(h_arr, d_arr);

  for (j = 0; j < lj0; j++) {
    h_ikj[j] = Complex(0.0, (2.0 * j * pi));
  }

  gt::backend::copy(h_ikj, d_ikj);

  // cpu reference
  for (int i = 0; i < time_warmup_count; i++) {
    ij_deriv_cpu(li0, lj0, lbg0, h_arr.data(), ncoeff, h_coeff.data(),
                 h_ikj.data(), ref_darr.data());
  }
  clock_gettime(CLOCK_MONOTONIC, &start);
  for (int i = 0; i < time_run_count; i++) {
    ij_deriv_cpu(li0, lj0, lbg0, h_arr.data(), ncoeff, h_coeff.data(),
                 h_ikj.data(), ref_darr.data());
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  seconds_per_run =
    ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1.0e-9) /
    time_run_count;
  printf("cpu     seconds/run: %0.6f\n", seconds_per_run);

#ifdef DEBUG_COMPARE
  array_stats(&minNorm, &maxNorm, &minRe, &maxRe, &minIm, &maxIm, h_arr.data(),
              arr_size);
  printf("arr  norm (%0.2e, %0.2e) re (%0.2e, %0.2e) im (%0.2e, %0.2e)\n",
         minNorm, maxNorm, minRe, maxRe, minIm, maxIm);
  array_stats(&minNorm, &maxNorm, &minRe, &maxRe, &minIm, &maxIm,
              ref_darr.data(), darr_size);
  printf("darr norm (%0.2e, %0.2e) re (%0.2e, %0.2e) im (%0.2e, %0.2e)\n",
         minNorm, maxNorm, minRe, maxRe, minIm, maxIm);
#endif

  // gtensor cpu
  for (int i = 0; i < time_warmup_count; i++) {
    ij_deriv_gt_host(li0, lj0, lbg0, h_arr.data(), ncoeff, h_coeff.data(),
                     h_ikj.data(), h_darr.data());
  }
  clock_gettime(CLOCK_MONOTONIC, &start);
  for (int i = 0; i < time_run_count; i++) {
    ij_deriv_gt_host(li0, lj0, lbg0, h_arr.data(), ncoeff, h_coeff.data(),
                     h_ikj.data(), h_darr.data());
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  seconds_per_run =
    ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1.0e-9) /
    time_run_count;
  printf("gt host seconds/run: %0.6f\n", seconds_per_run);

#ifdef DEBUG_COMPARE
  gt::backend::copy(d_darr, h_darr);
  compare_deriv(&error, &maxError, &relError, &maxRelError, ref_darr.data(),
                h_darr.data(), 0, li0, 0, lj0, 0, lbg0, 0, li0, lj0, lbg0, 2);
  printf("gt host diff[x]: %0.4e (max %0.4e) | rel %0.4e (max %0.4e)\n", error,
         maxError, relError, maxRelError);
  compare_deriv(&error, &maxError, &relError, &maxRelError, ref_darr.data(),
                h_darr.data(), 0, li0, 0, lj0, 0, lbg0, 1, li0, lj0, lbg0, 2);
  printf("gt host diff[y]: %0.4e (max %0.4e) | rel %0.4e (max %0.4e)\n", error,
         maxError, relError, maxRelError);
#endif

  // native GPU api
  for (int i = 0; i < time_warmup_count; i++) {
    ij_deriv_gpu(li0, lj0, lbg0, d_arr.data(), ncoeff, d_coeff.data(),
                 d_ikj.data(), d_darr.data());
  }
  clock_gettime(CLOCK_MONOTONIC, &start);
  for (int i = 0; i < time_run_count; i++) {
    ij_deriv_gpu(li0, lj0, lbg0, d_arr.data(), ncoeff, d_coeff.data(),
                 d_ikj.data(), d_darr.data());
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  seconds_per_run =
    ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1.0e-9) /
    time_run_count;
  printf("gpu     seconds/run: %0.6f\n", seconds_per_run);

#ifdef DEBUG_COMPARE
  gt::backend::copy(d_darr, h_darr);
  compare_deriv(&error, &maxError, &relError, &maxRelError, ref_darr.data(),
                h_darr.data(), 0, li0, 0, lj0, 0, lbg0, 0, li0, lj0, lbg0, 2);
  printf("gpu diff[x]: %0.4e (max %0.4e) | rel %0.4e (max %0.4e)\n", error,
         maxError, relError, maxRelError);
  compare_deriv(&error, &maxError, &relError, &maxRelError, ref_darr.data(),
                h_darr.data(), 0, li0, 0, lj0, 0, lbg0, 1, li0, lj0, lbg0, 2);
  printf("gpu diff[y]: %0.4e (max %0.4e) | rel %0.4e (max %0.4e)\n", error,
         maxError, relError, maxRelError);
#endif

  // gtensor gpu
  for (int i = 0; i < time_warmup_count; i++) {
    ij_deriv_gt_device(li0, lj0, lbg0, d_arr.data(), ncoeff, h_coeff.data(),
                       d_ikj.data(), d_darr.data());
  }
  clock_gettime(CLOCK_MONOTONIC, &start);
  for (int i = 0; i < time_run_count; i++) {
    ij_deriv_gt_device(li0, lj0, lbg0, d_arr.data(), ncoeff, h_coeff.data(),
                       d_ikj.data(), d_darr.data());
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  seconds_per_run =
    ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1.0e-9) /
    time_run_count;
  printf("gt gpu  seconds/run: %0.6f\n", seconds_per_run);

#ifdef DEBUG_COMPARE
  gt::backend::copy(d_darr, h_darr);
  compare_deriv(&error, &maxError, &relError, &maxRelError, ref_darr.data(),
                h_darr.data(), 0, li0, 0, lj0, 0, lbg0, 0, li0, lj0, lbg0, 2);
  printf("gt  diff[x]: %0.4e (max %0.4e) | rel %0.4e (max %0.4e)\n", error,
         maxError, relError, maxRelError);
  compare_deriv(&error, &maxError, &relError, &maxRelError, ref_darr.data(),
                h_darr.data(), 0, li0, 0, lj0, 0, lbg0, 1, li0, lj0, lbg0, 2);
  printf("gt  diff[y]: %0.4e (max %0.4e) | rel %0.4e (max %0.4e)\n", error,
         maxError, relError, maxRelError);
#endif
}

int main(int argc, char** argv)
{
  test_ij_deriv<double>(1024, 16, 32);
  test_ij_deriv<double>(1024, 32, 32);
  test_ij_deriv<double>(1024, 64, 32);
  test_ij_deriv<double>(1024, 128, 32);
  test_ij_deriv<double>(1024, 256, 32);
  test_ij_deriv<double>(1024, 256, 64);
}
