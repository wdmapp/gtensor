#ifndef GTENSOR_FFT_BACKEND_HOST_H
#define GTENSOR_FFT_BACKEND_HOST_H

#include <numeric>
#include <stdexcept>
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

namespace detail
{

template <gt::fft::Domain D, typename R>
struct fft_config;

template <>
struct fft_config<gt::fft::Domain::COMPLEX, double>
{
  using Tin = gt::complex<double>;
  using Tout = gt::complex<double>;
  using Bin = fftw_complex;
  using Bout = fftw_complex;
};

template <>
struct fft_config<gt::fft::Domain::COMPLEX, float>
{
  using Tin = gt::complex<float>;
  using Tout = gt::complex<float>;
  using Bin = fftwf_complex;
  using Bout = fftwf_complex;
};

template <>
struct fft_config<gt::fft::Domain::REAL, double>
{
  using Tin = double;
  using Tout = gt::complex<double>;
  using Bin = double;
  using Bout = fftw_complex;
};

template <>
struct fft_config<gt::fft::Domain::REAL, float>
{
  using Tin = float;
  using Tout = gt::complex<float>;
  using Bin = float;
  using Bout = fftwf_complex;
};

} // namespace detail

template <gt::fft::Domain D, typename R>
class FFTPlanManyHost
{
  using fftw = fftw::fftw<R>;
  using Bin = typename detail::fft_config<D, R>::Bin;
  using Bout = typename detail::fft_config<D, R>::Bout;

public:
  FFTPlanManyHost(std::vector<int> lengths, int batch_size = 1,
                  gt::stream_view stream = gt::stream_view{})
    : is_valid_(true)
  {
    int rank = lengths.size();
    int idist, odist;
    idist = std::accumulate(lengths.begin(), lengths.end(), 1,
                            std::multiplies<int>());
    if (D == gt::fft::Domain::REAL) {
      odist = idist / lengths[rank - 1] * (lengths[rank - 1] / 2 + 1);
    } else {
      odist = idist;
    }
    init(lengths, 1, idist, 1, odist, batch_size);
  }

  FFTPlanManyHost(std::vector<int> lengths, int istride, int idist, int ostride,
                  int odist, int batch_size = 1,
                  gt::stream_view stream = gt::stream_view{})
    : is_valid_(true)
  {
    init(lengths, istride, idist, ostride, odist, batch_size);
  }

  // move only
  // delete copy ctor/assign
  FFTPlanManyHost(const FFTPlanManyHost& other) = delete;
  FFTPlanManyHost& operator=(const FFTPlanManyHost& other) = delete;

  // custom move to avoid double destroy in moved-from object
  FFTPlanManyHost(FFTPlanManyHost&& other) : is_valid_(true)
  {
    fftw_forward_ = other.fftw_forward_;
    fftw_inverse_ = other.fftw_inverse_;
    other.is_valid_ = false;
  }

  FFTPlanManyHost& operator=(FFTPlanManyHost&& other)
  {
    fftw_forward_ = other.fftw_forward_;
    fftw_inverse_ = other.fftw_inverse_;
    other.is_valid_ = false;
    return *this;
  }

  virtual ~FFTPlanManyHost()
  {
    if (is_valid_) {
      // fftw_destroy_plan(fftw_forward_);
      // fftw_destroy_plan(fftw_inverse_);
    }
  }

  void operator()(typename detail::fft_config<D, R>::Tin* indata,
                  typename detail::fft_config<D, R>::Tout* outdata) const
  {
    if (!is_valid_) {
      throw std::runtime_error("can't use a moved-from plan");
    }
    auto bin = reinterpret_cast<Bin*>(indata);
    auto bout = reinterpret_cast<Bout*>(outdata);
    if constexpr (D == gt::fft::Domain::COMPLEX) {
      fftw_forward_.execute_dft(bin, bout);
    }
  }

  void inverse(typename detail::fft_config<D, R>::Tout* indata,
               typename detail::fft_config<D, R>::Tin* outdata) const
  {
    if (!is_valid_) {
      throw std::runtime_error("can't use a moved-from plan");
    }
    auto bin = reinterpret_cast<Bout*>(indata);
    auto bout = reinterpret_cast<Bin*>(outdata);
    if constexpr (D == gt::fft::Domain::COMPLEX) {
      fftw_inverse_.execute_dft(bin, bout);
    }
  }

  std::size_t get_work_buffer_bytes() { return 0; }

private:
  void init(std::vector<int> lengths, int istride, int idist, int ostride,
            int odist, int batch_size)
  {
    int rank = lengths.size();
    int* n = lengths.data();

    if constexpr (D == gt::fft::Domain::COMPLEX) {
      using Bin = typename detail::fft_config<D, R>::Bin;
      using Bout = typename detail::fft_config<D, R>::Bout;
      Bin* in = nullptr;
      Bout* out = nullptr;
      fftw_forward_ =
        fftw::plan_many_dft(rank, n, batch_size, in, NULL, istride, idist, out,
                            NULL, ostride, odist, -1, FFTW_ESTIMATE);
      fftw_inverse_ =
        fftw::plan_many_dft(rank, n, batch_size, out, NULL, ostride, odist, in,
                            NULL, istride, idist, 1, FFTW_ESTIMATE);
    }
  }

  fftw fftw_inverse_;
  fftw fftw_forward_;
  bool is_valid_;
};

template <gt::fft::Domain D, typename R>
using FFTPlanManyBackend = FFTPlanManyHost<D, R>;

} // namespace fft

} // namespace gt

#endif // GTENSOR_FFT_CUDA_H
