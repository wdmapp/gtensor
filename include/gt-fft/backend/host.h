#ifndef GTENSOR_FFT_BACKEND_HOST_H
#define GTENSOR_FFT_BACKEND_HOST_H

#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include <fftw3.h>

namespace gt
{

namespace fftw
{
template <typename R>
class fftw;

#define R double
#define FFTW_(SFX) fftw_##SFX
#include "fftw_inc.h"
#undef R
#undef FFTW_

#define R float
#define FFTW_(SFX) fftwf_##SFX
#include "fftw_inc.h"
#undef R
#undef FFTW_

} // namespace fftw

namespace fft
{

template <gt::fft::Domain D, typename R>
class FFTPlanManyHost;

template <typename R>
class FFTPlanManyHost<gt::fft::Domain::COMPLEX, R>
{
  using fftw_type = fftw::fftw<R>;
  using complex_type = gt::complex<R>;
  using fftw_complex_type = typename fftw_type::complex_type;

public:
  FFTPlanManyHost(std::vector<int> lengths, int batch_size = 1,
                  gt::stream_view stream = gt::stream_view{})
  {
    int rank = lengths.size();
    int idist = std::accumulate(lengths.begin(), lengths.end(), 1,
                                std::multiplies<int>());
    int odist = idist;
    init(lengths, 1, idist, 1, odist, batch_size);
  }

  FFTPlanManyHost(std::vector<int> lengths, int istride, int idist, int ostride,
                  int odist, int batch_size = 1,
                  gt::stream_view stream = gt::stream_view{})
  {
    init(lengths, istride, idist, ostride, odist, batch_size);
  }

  void operator()(complex_type* indata, complex_type* outdata) const
  {
    fftw_forward_.execute_dft(reinterpret_cast<fftw_complex_type*>(indata),
                              reinterpret_cast<fftw_complex_type*>(outdata));
  }

  void inverse(complex_type* indata, complex_type* outdata) const
  {
    fftw_inverse_.execute_dft(reinterpret_cast<fftw_complex_type*>(indata),
                              reinterpret_cast<fftw_complex_type*>(outdata));
  }

  std::size_t get_work_buffer_bytes() { return 0; }

private:
  void init(std::vector<int> lengths, int istride, int idist, int ostride,
            int odist, int batch_size)
  {
    int rank = lengths.size();
    int* n = lengths.data();
    fftw_complex_type dummy_in, dummy_out;

    fftw_forward_ = fftw_type::plan_many_dft(
      rank, n, batch_size, &dummy_in, NULL, istride, idist, &dummy_out, NULL,
      ostride, odist, -1, FFTW_ESTIMATE);
    fftw_inverse_ = fftw_type::plan_many_dft(
      rank, n, batch_size, &dummy_out, NULL, ostride, odist, &dummy_in, NULL,
      istride, idist, 1, FFTW_ESTIMATE);
  }

  fftw_type fftw_forward_;
  fftw_type fftw_inverse_;
};

template <typename R>
class FFTPlanManyHost<gt::fft::Domain::REAL, R>
{
  using fftw_type = fftw::fftw<R>;
  using complex_type = gt::complex<R>;
  using real_type = R;
  using fftw_complex_type = typename fftw_type::complex_type;
  using fftw_real_type = typename fftw_type::real_type;

public:
  FFTPlanManyHost(std::vector<int> lengths, int batch_size = 1,
                  gt::stream_view stream = gt::stream_view{})
  {
    int rank = lengths.size();
    int idist = std::accumulate(lengths.begin(), lengths.end(), 1,
                                std::multiplies<int>());
    int odist = idist / lengths[rank - 1] * (lengths[rank - 1] / 2 + 1);
    init(lengths, 1, idist, 1, odist, batch_size);
  }

  FFTPlanManyHost(std::vector<int> lengths, int istride, int idist, int ostride,
                  int odist, int batch_size = 1,
                  gt::stream_view stream = gt::stream_view{})
  {
    init(lengths, istride, idist, ostride, odist, batch_size);
  }

  void operator()(real_type* indata, complex_type* outdata) const
  {
    auto bin = reinterpret_cast<fftw_real_type*>(indata);
    auto bout = reinterpret_cast<fftw_complex_type*>(outdata);
    fftw_forward_.execute_dft_r2c(bin, bout);
  }

  void inverse(complex_type* indata, real_type* outdata) const
  {
    auto bin = reinterpret_cast<fftw_complex_type*>(indata);
    auto bout = reinterpret_cast<fftw_real_type*>(outdata);
    fftw_inverse_.execute_dft_c2r(bin, bout);
  }

  std::size_t get_work_buffer_bytes() { return 0; }

private:
  void init(std::vector<int> lengths, int istride, int idist, int ostride,
            int odist, int batch_size)
  {
    int rank = lengths.size();
    int* n = lengths.data();
    fftw_real_type dummy_in;
    fftw_complex_type dummy_out;

    fftw_forward_ = fftw_type::plan_many_dft_r2c(
      rank, n, batch_size, &dummy_in, NULL, istride, idist, &dummy_out, NULL,
      ostride, odist, FFTW_ESTIMATE);
    fftw_inverse_ = fftw_type::plan_many_dft_c2r(
      rank, n, batch_size, &dummy_out, NULL, ostride, odist, &dummy_in, NULL,
      istride, idist, FFTW_ESTIMATE);
  }

  fftw_type fftw_forward_;
  fftw_type fftw_inverse_;
};

template <gt::fft::Domain D, typename R>
using FFTPlanManyBackend = FFTPlanManyHost<D, R>;

} // namespace fft

} // namespace gt

#endif // GTENSOR_FFT_BACKEND_HOST_H
