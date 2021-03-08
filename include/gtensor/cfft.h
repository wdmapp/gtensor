#include "gtensor/fft.h"

#ifdef __cplusplus
extern "C" {
#endif

gt::fft::FFTPlanMany<gt::fft::Domain::COMPLEX, float>* gtfft_new_complex_float(
  int rank, int* n, int istride, int idist, int ostride, int odist,
  int batch_size);

gt::fft::FFTPlanMany<gt::fft::Domain::COMPLEX, double>*
gtfft_new_complex_double(int rank, int* n, int istride, int idist, int ostride,
                         int odist, int batch_size);

gt::fft::FFTPlanMany<gt::fft::Domain::REAL, float>* gtfft_new_real_float(
  int rank, int* n, int istride, int idist, int ostride, int odist,
  int batch_size);

gt::fft::FFTPlanMany<gt::fft::Domain::REAL, double>* gtfft_new_real_double(
  int rank, int* n, int istride, int idist, int ostride, int odist,
  int batch_size);

void gtfft_delete_complex_float(
  gt::fft::FFTPlanMany<gt::fft::Domain::COMPLEX, float>* plan);

void gtfft_delete_complex_double(
  gt::fft::FFTPlanMany<gt::fft::Domain::COMPLEX, double>* plan);

void gtfft_delete_real_float(
  gt::fft::FFTPlanMany<gt::fft::Domain::REAL, float>* plan);

void gtfft_delete_real_double(
  gt::fft::FFTPlanMany<gt::fft::Domain::REAL, double>* plan);

void gtfft_zz(gt::fft::FFTPlanMany<gt::fft::Domain::COMPLEX, double>* plan,
              gt::complex<double>* indata, gt::complex<double>* outdata);
void gtfft_inverse_zz(
  gt::fft::FFTPlanMany<gt::fft::Domain::COMPLEX, double>* plan,
  gt::complex<double>* indata, gt::complex<double>* outdata);

void gtfft_cc(gt::fft::FFTPlanMany<gt::fft::Domain::COMPLEX, float>* plan,
              gt::complex<float>* indata, gt::complex<float>* outdata);
void gtfft_inverse_cc(
  gt::fft::FFTPlanMany<gt::fft::Domain::COMPLEX, float>* plan,
  gt::complex<float>* indata, gt::complex<float>* outdata);

void gtfft_dz(gt::fft::FFTPlanMany<gt::fft::Domain::REAL, double>* plan,
              double* indata, gt::complex<double>* outdata);
void gtfft_inverse_zd(gt::fft::FFTPlanMany<gt::fft::Domain::REAL, double>* plan,
                      gt::complex<double>* indata, double* outdata);

void gtfft_rc(gt::fft::FFTPlanMany<gt::fft::Domain::REAL, float>* plan,
              float* indata, gt::complex<float>* outdata);
void gtfft_inverse_cr(gt::fft::FFTPlanMany<gt::fft::Domain::REAL, float>* plan,
                      gt::complex<float>* indata, float* outdata);

#ifdef __cplusplus
}
#endif
