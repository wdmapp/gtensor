#ifndef GTENSOR_FFT_BBFFT_H
#define GTENSOR_FFT_BBFFT_H

#include <memory>
#include <numeric>
#include <stdexcept>

#include "gtensor/backend_sycl.h"
#include "gtensor/complex.h"

#include "bbfft/sycl/make_plan.hpp"

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
  using Tin = gt::complex<double>;
  using Tout = gt::complex<double>;
  using Bin = std::complex<double>;
  using Bout = std::complex<double>;
  static const bbfft::transform_type transform_forward =
    bbfft::transform_type::c2c;
  static const bbfft::transform_type transform_inverse =
    bbfft::transform_type::c2c;
};

template <>
struct fft_config<gt::fft::Domain::COMPLEX, float>
{
  using Tin = gt::complex<float>;
  using Tout = gt::complex<float>;
  using Bin = std::complex<float>;
  using Bout = std::complex<float>;
  static const bbfft::transform_type transform_forward =
    bbfft::transform_type::c2c;
  static const bbfft::transform_type transform_inverse =
    bbfft::transform_type::c2c;
};

template <>
struct fft_config<gt::fft::Domain::REAL, double>
{
  using Tin = double;
  using Tout = gt::complex<double>;
  using Bin = double;
  using Bout = std::complex<double>;
  static const bbfft::transform_type transform_forward =
    bbfft::transform_type::r2c;
  static const bbfft::transform_type transform_inverse =
    bbfft::transform_type::c2r;
};

template <>
struct fft_config<gt::fft::Domain::REAL, float>
{
  using Tin = float;
  using Tout = gt::complex<float>;
  using Bin = float;
  using Bout = std::complex<float>;
  static const bbfft::transform_type transform_forward =
    bbfft::transform_type::r2c;
  static const bbfft::transform_type transform_inverse =
    bbfft::transform_type::c2r;
};

} // namespace detail

template <gt::fft::Domain D, typename R>
class FFTPlanManyBBFFT
{
  using Config = typename detail::fft_config<D, R>;

public:
  FFTPlanManyBBFFT(std::vector<int> lengths, int batch_size = 1,
                   gt::stream_view stream = gt::stream_view{})
  {
    init(lengths, 1, 0, 1, 0, batch_size, stream);
  }

  FFTPlanManyBBFFT(std::vector<int> lengths, int istride, int idist,
                   int ostride, int odist, int batch_size = 1,
                   gt::stream_view stream = gt::stream_view{})
  {
    init(lengths, istride, idist, ostride, odist, batch_size, stream);
  }

  // move only
  FFTPlanManyBBFFT(FFTPlanManyBBFFT&& other) = default;
  FFTPlanManyBBFFT& operator=(FFTPlanManyBBFFT&& other) = default;

  // delete copy ctor/assign
  FFTPlanManyBBFFT(const FFTPlanManyBBFFT& other) = delete;
  FFTPlanManyBBFFT& operator=(const FFTPlanManyBBFFT& other) = delete;

  void operator()(typename detail::fft_config<D, R>::Tin* indata,
                  typename detail::fft_config<D, R>::Tout* outdata) const
  {
    if (!plan_forward_) {
      throw std::runtime_error("can't use a moved-from plan");
    }
    using Bin = typename detail::fft_config<D, R>::Bin;
    using Bout = typename detail::fft_config<D, R>::Bout;
    auto bin = reinterpret_cast<Bin*>(indata);
    auto bout = reinterpret_cast<Bout*>(outdata);
    auto e = plan_forward_.execute(bin, bout);
    e.wait();
  }

  void inverse(typename detail::fft_config<D, R>::Tout* indata,
               typename detail::fft_config<D, R>::Tin* outdata) const
  {
    if (!plan_forward_) {
      throw std::runtime_error("can't use a moved-from plan");
    }
    using Breal = typename detail::fft_config<D, R>::Bin;
    using Bcmplx = typename detail::fft_config<D, R>::Bout;
    auto bin = reinterpret_cast<Bcmplx*>(indata);
    auto bout = reinterpret_cast<Breal*>(outdata);
    auto e = plan_inverse_.execute(bin, bout);
  }

private:
  void init(std::vector<int> lengths_, int istride, int idist, int ostride,
            int odist, int batch_size, gt::stream_view stream)
  {
    auto& q = stream.get_backend_stream();

    unsigned int rank = lengths_.size();

    bool layout_supported = true;

    if (rank > 1 && (istride != 1 || ostride != 1)) {
      layout_supported = false;
    }

    // shape is { M, N_1, N_2, N_3, K }. We always use M=1, K is number of
    // batches, and the N's are the dimensions from length vector arg
    std::array<std::size_t, bbfft::max_tensor_dim> shape;
    shape[0] = 1;
    for (int i = 1; i <= rank; i++) {
      // shape is reverse of lengths, column major vs row major
      shape[i] = lengths_[rank - i];
    }
    shape[rank + 1] = batch_size;
    for (int i = rank + 2; i < bbfft::max_tensor_dim; i++) {
      shape[i] = 0;
    }

    std::size_t cshape1;
    std::size_t istrideu = istride;
    std::size_t ostrideu = ostride;
    if constexpr (D == gt::fft::Domain::REAL) {
      cshape1 = shape[1] / 2 + 1;
    } else {
      cshape1 = shape[1];
    }
    std::array<std::size_t, bbfft::max_tensor_dim> rstride = {
      1u, istrideu, istrideu * shape[1]};
    std::array<std::size_t, bbfft::max_tensor_dim> cstride = {
      1u, ostrideu, ostrideu * cshape1};

    for (int i = 1; i < rank; i++) {
      rstride[i + 2] = shape[i + 1] * rstride[i + 1];
      cstride[i + 2] = shape[i + 1] * cstride[i + 1];
    }

    if (idist != 0) {
      if (rank > 1 && idist != rstride[rank + 1]) {
        layout_supported = false;
      }
      rstride[rank + 1] = idist;
    }
    if (odist != 0) {
      if (rank > 1 && odist != cstride[rank + 1]) {
        layout_supported = false;
      }
      cstride[rank + 1] = odist;
    }

    if (!layout_supported) {
      throw std::runtime_error(
        "bbft backend requires standard layout for 2d/3d fft");
    }

    bbfft::configuration cfg_fwd = {
      rank,                      // dim
      shape,                     // { M, N_1, N_2, ..., K }
      bbfft::to_precision_v<R>,  // precision
      bbfft::direction::forward, // direction
      Config::transform_forward, // type
      rstride,                   // input strides
      cstride                    // output stride
    };
    // cfg_fwd.set_strides_default(false);
    bbfft::configuration cfg_inv = {rank,
                                    shape,
                                    bbfft::to_precision_v<R>,
                                    bbfft::direction::backward,
                                    Config::transform_inverse,
                                    cstride,
                                    rstride};
    // cfg_inv.set_strides_default(false);

    plan_forward_ = bbfft::make_plan(cfg_fwd, q);
    plan_inverse_ = bbfft::make_plan(cfg_inv, q);
  }

  mutable bbfft::plan<::sycl::event> plan_forward_;
  mutable bbfft::plan<::sycl::event> plan_inverse_;
};

template <gt::fft::Domain D, typename R>
using FFTPlanManyBackend = FFTPlanManyBBFFT<D, R>;

} // namespace fft

} // namespace gt

#endif // GTENSOR_FFT_BBFFT_H
