#ifndef GTENSOR_FFT_SYCL_H
#define GTENSOR_FFT_SYCL_H

#include <memory>
#include <numeric>
#include <stdexcept>

#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>

#include "gtensor/backend_sycl.h"

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
  FFTPlanManySYCL(std::vector<int> lengths, int batch_size = 1)
  {
    init(lengths, 1, 0, 1, 0, batch_size);
  }

  FFTPlanManySYCL(std::vector<int> lengths, int istride, int idist, int ostride,
                  int odist, int batch_size = 1)
  {
    init(lengths, istride, idist, ostride, odist, batch_size);
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
    if (plan_forward_ == nullptr) {
      throw std::runtime_error("can't use a moved-from plan");
    }
    using Bin = typename detail::fft_config<D, R>::Bin;
    using Bout = typename detail::fft_config<D, R>::Bout;
    auto bin = reinterpret_cast<Bin*>(indata);
    auto bout = reinterpret_cast<Bout*>(outdata);
    auto e = oneapi::mkl::dft::compute_forward(*plan_forward_, bin, bout);
    e.wait();
  }

  void inverse(typename detail::fft_config<D, R>::Tout* indata,
               typename detail::fft_config<D, R>::Tin* outdata) const
  {
    if (plan_forward_ == nullptr) {
      throw std::runtime_error("can't use a moved-from plan");
    }
    using Breal = typename detail::fft_config<D, R>::Bin;
    using Bcmplx = typename detail::fft_config<D, R>::Bout;
    auto bin = reinterpret_cast<Bcmplx*>(indata);
    auto bout = reinterpret_cast<Breal*>(outdata);
    if (is_layout_asymmetric_) {
      oneapi::mkl::dft::compute_backward(*plan_inverse_, bin, bout).wait();
    } else {
      oneapi::mkl::dft::compute_backward(*plan_forward_, bin, bout).wait();
    }
  }

private:
  void init(std::vector<int> lengths_, int istride, int idist, int ostride,
            int odist, int batch_size = 1)
  {
    int rank = lengths_.size();

    std::vector<MKL_FFT_LONG> lengths(rank);
    std::vector<MKL_FFT_LONG> freq_lengths(rank);
    for (int i = 0; i < rank; i++) {
      lengths[i] = lengths_[i];
      if (D == gt::fft::Domain::REAL && i == rank - 1) {
        freq_lengths[i] = lengths_[i] / 2 + 1;
      } else {
        freq_lengths[i] = lengths_[i];
      }
    }

    if (idist == 0) {
      idist = std::accumulate(lengths.begin(), lengths.end(), 1,
                              std::multiplies<MKL_FFT_LONG>());
    }
    if (odist == 0) {
      odist = std::accumulate(freq_lengths.begin(), freq_lengths.end(), 1,
                              std::multiplies<MKL_FFT_LONG>());
    }

    // Note: COMPLEX asymmetric transforms still auto switch in/out strides
    // for rank 1, which is an inconsistency.
    if (D == gt::fft::Domain::REAL) {
      is_layout_asymmetric_ = true;
    } else if (istride != ostride && rank > 1) {
      is_layout_asymmetric_ = true;
    } else {
      is_layout_asymmetric_ = false;
    }

    try {
      if (rank > 1) {
        plan_forward_ = std::make_unique<Desc>(lengths);
        if (is_layout_asymmetric_) {
          plan_inverse_ = std::make_unique<Desc>(lengths);
        }
      } else {
        plan_forward_ = std::make_unique<Desc>(lengths[0]);
        if (is_layout_asymmetric_) {
          plan_inverse_ = std::make_unique<Desc>(lengths[0]);
        }
      }

      // Set up strides arrays used to map multi-d indexing
      // to 1d. Fastest changing index is in right most
      // position.
      std::int64_t rstrides[rank + 1];
      std::int64_t cstrides[rank + 1];
      rstrides[0] = 0;
      cstrides[0] = 0;
      std::int64_t rs = istride;
      std::int64_t cs = ostride;
      for (int i = rank; i > 0; i--) {
        rstrides[i] = rs;
        cstrides[i] = cs;
        rs *= lengths[i - 1];
        cs *= freq_lengths[i - 1];
      }

      plan_forward_->set_value(
        oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, batch_size);
      plan_forward_->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                               rstrides);
      plan_forward_->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                               cstrides);
      plan_forward_->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                               idist);
      plan_forward_->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                               odist);
      plan_forward_->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                               DFTI_NOT_INPLACE);
      plan_forward_->set_value(
        oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE,
        DFTI_COMPLEX_COMPLEX);
      plan_forward_->commit(gt::backend::sycl::get_queue());

      if (is_layout_asymmetric_) {
        plan_inverse_->set_value(
          oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, batch_size);
        plan_inverse_->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                                 cstrides);
        plan_inverse_->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                                 rstrides);
        plan_inverse_->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                                 idist);
        plan_inverse_->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                                 odist);
        plan_inverse_->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                                 DFTI_NOT_INPLACE);
        plan_inverse_->set_value(
          oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE,
          DFTI_COMPLEX_COMPLEX);
        plan_inverse_->commit(gt::backend::sycl::get_queue());
      }
    } catch (std::exception const& e) {
      std::cerr << "Error creating dft descriptor:" << e.what() << std::endl;
      abort();
    }
  }

  std::unique_ptr<Desc> plan_forward_;
  std::unique_ptr<Desc> plan_inverse_;

  // flag to keep track of whether a separate inverse plan is required
  bool is_layout_asymmetric_;
};

template <gt::fft::Domain D, typename R>
using FFTPlanManyBackend = FFTPlanManySYCL<D, R>;

} // namespace fft

} // namespace gt

#endif // GTENSOR_FFT_SYCL_H
