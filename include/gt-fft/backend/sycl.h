#ifndef GTENSOR_FFT_SYCL_H
#define GTENSOR_FFT_SYCL_H

#include <memory>
#include <stdexcept>

#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>

#include "gtensor/sycl_backend.h"

// Should be possible to use MKL_LONG, but for somereason it is not defined
// correctly even when MKL_ILP64 is set correctly.
#define MKL_FFT_LONG std::int64_t

namespace gt
{

namespace fft
{

namespace detail
{

template <gt::fft::Domain D, typename R>
struct fft_config;

template <>
struct fft_config<gt::fft::Domain::COMPLEX, double>
{
  using Desc = oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
                                            oneapi::mkl::dft::domain::COMPLEX>;
  using Tin = gt::complex<double>;
  using Tout = gt::complex<double>;
  using Bin = gt::complex<double>;
  using Bout = gt::complex<double>;
};

template <>
struct fft_config<gt::fft::Domain::COMPLEX, float>
{
  using Desc = oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE,
                                            oneapi::mkl::dft::domain::COMPLEX>;
  using Tin = gt::complex<float>;
  using Tout = gt::complex<float>;
  using Bin = gt::complex<float>;
  using Bout = gt::complex<float>;
};

template <>
struct fft_config<gt::fft::Domain::REAL, double>
{
  using Desc = oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
                                            oneapi::mkl::dft::domain::REAL>;
  using Tin = double;
  using Tout = gt::complex<double>;
  using Bin = double;
  using Bout = double;
};

template <>
struct fft_config<gt::fft::Domain::REAL, float>
{
  using Desc = oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE,
                                            oneapi::mkl::dft::domain::REAL>;
  using Tin = float;
  using Tout = gt::complex<float>;
  using Bin = float;
  using Bout = float;
};

} // namespace detail

template <gt::fft::Domain D, typename R>
class FFTPlanManySYCL
{
  using Desc = typename detail::fft_config<D, R>::Desc;

public:
  FFTPlanManySYCL(std::vector<MKL_FFT_LONG> lengths, int batch_size = 1)
  {
    MKL_FFT_LONG fwd_distance, bwd_distance;

    int rank = lengths.size();

    fwd_distance = std::accumulate(lengths.begin(), lengths.end(), 1,
                                   std::multiplies<MKL_FFT_LONG>());
    if (D == gt::fft::Domain::REAL) {
      bwd_distance =
        fwd_distance / lengths[rank - 1] * (lengths[rank - 1] / 2 + 1);
    } else {
      bwd_distance = fwd_distance;
    }

    try {
      if (rank > 1) {
        plan_ = std::make_unique<Desc>(lengths);
      } else {
        plan_ = std::make_unique<Desc>(lengths[0]);
      }

      // set up strides arrays
      std::int64_t rstrides[rank + 1];
      std::int64_t cstrides[rank + 1];
      rstrides[0] = 0;
      cstrides[0] = 0;
      std::int64_t rs = 1;
      std::int64_t cs = 1;
      for (int i = 1; i <= rank; i++) {
        rstrides[i] = rs;
        cstrides[i] = cs;
        rs *= lengths[i - 1];
        cs *= lengths[i - 1];
      }

      plan_->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                       batch_size);
      plan_->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, rstrides);
      plan_->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                       cstrides);
      plan_->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                       fwd_distance);
      plan_->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                       bwd_distance);
      plan_->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                       DFTI_NOT_INPLACE);
      plan_->set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE,
                       DFTI_COMPLEX_COMPLEX);
      plan_->set_value(oneapi::mkl::dft::config_param::FORWARD_SCALE, R(1));
      plan_->set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE, R(1));
      plan_->commit(gt::backend::sycl::get_queue());
    } catch (std::exception const& e) {
      std::cerr << "Error creating dft descriptor:" << e.what() << std::endl;
      abort();
    }
  }

  // move only
  // delete copy ctor/assign
  FFTPlanManySYCL(const FFTPlanManySYCL& other) = delete;
  FFTPlanManySYCL& operator=(const FFTPlanManySYCL& other) = delete;

  // default move ctor/assign
  FFTPlanManySYCL(FFTPlanManySYCL&& other) = default;
  FFTPlanManySYCL& operator=(FFTPlanManySYCL&& other) = default;

  void operator()(typename detail::fft_config<D, R>::Tin* indata,
                  typename detail::fft_config<D, R>::Tout* outdata) const
  {
    if (plan_ == nullptr) {
      throw std::runtime_error("can't use a moved-from plan");
    }
    using Bin = typename detail::fft_config<D, R>::Bin;
    using Bout = typename detail::fft_config<D, R>::Bout;
    auto bin = reinterpret_cast<Bin*>(indata);
    auto bout = reinterpret_cast<Bout*>(outdata);
    auto e = oneapi::mkl::dft::compute_forward(*plan_, bin, bout);
    e.wait();
  }

  void inverse(typename detail::fft_config<D, R>::Tout* indata,
               typename detail::fft_config<D, R>::Tin* outdata) const
  {
    if (plan_ == nullptr) {
      throw std::runtime_error("can't use a moved-from plan");
    }
    using Bin = typename detail::fft_config<D, R>::Bin;
    using Bout = typename detail::fft_config<D, R>::Bout;
    auto bin = reinterpret_cast<Bout*>(indata);
    auto bout = reinterpret_cast<Bin*>(outdata);
    auto e = oneapi::mkl::dft::compute_backward(*plan_, bin, bout);
    e.wait();
  }

private:
  std::unique_ptr<Desc> plan_;
};

template <gt::fft::Domain D, typename R>
using FFTPlanManyBackend = FFTPlanManySYCL<D, R>;

} // namespace fft

} // namespace gt

#endif // GTENSOR_FFT_SYCL_H
