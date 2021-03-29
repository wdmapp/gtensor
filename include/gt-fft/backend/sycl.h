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
  FFTPlanManySYCL(int rank, int* n, int istride, int idist, int ostride,
                  int odist, int batch_size)
  {
    MKL_FFT_LONG fwd_distance, bwd_distance;

    fwd_distance = idist;
    bwd_distance = odist;

    try {
      if (rank > 1) {
        std::vector<MKL_FFT_LONG> dims(rank);
        for (int i = 0; i < rank; i++) {
          dims[i] = n[i];
        }
        assert(dims.size() == rank);
        plan_ = std::make_unique<Desc>(dims);
      } else {
        plan_ = std::make_unique<Desc>(n[0]);
      }

      // set up strides arrays
      // TODO: is this correct column major?
      std::int64_t rstrides[rank + 1];
      std::int64_t cstrides[rank + 1];
      rstrides[0] = 0;
      cstrides[0] = 0;
      std::int64_t rs = istride;
      std::int64_t cs = ostride;
      for (int i = 1; i <= rank; i++) {
        rstrides[i] = rs;
        cstrides[i] = cs;
        rs *= n[i - 1];
        cs *= n[i - 1];
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

  void operator()(const typename detail::fft_config<D, R>::Tin* indata,
                  typename detail::fft_config<D, R>::Tout* outdata) const
  {
    if (plan_ == nullptr) {
      throw std::runtime_error("can't use a moved-from plan");
    }
    using Tin = typename detail::fft_config<D, R>::Tin;
    using Bin = typename detail::fft_config<D, R>::Bin;
    using Bout = typename detail::fft_config<D, R>::Bout;
    auto bin = reinterpret_cast<Bin*>(const_cast<Tin*>(indata));
    auto bout = reinterpret_cast<Bout*>(outdata);
    auto e = oneapi::mkl::dft::compute_forward(*plan_, bin, bout);
    e.wait();
  }

  void inverse(const typename detail::fft_config<D, R>::Tout* indata,
               typename detail::fft_config<D, R>::Tin* outdata) const
  {
    if (plan_ == nullptr) {
      throw std::runtime_error("can't use a moved-from plan");
    }
    using Tout = typename detail::fft_config<D, R>::Tout;
    using Bin = typename detail::fft_config<D, R>::Bin;
    using Bout = typename detail::fft_config<D, R>::Bout;
    auto bin = reinterpret_cast<Bout*>(const_cast<Tout*>(indata));
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
