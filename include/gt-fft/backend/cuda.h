#ifndef GTENSOR_FFT_CUDA_H
#define GTENSOR_FFT_CUDA_H

#include <numeric>
#include <stdexcept>
#include <vector>

// Note: this file is included by fft/hip.h after redef/type aliasing
// all the necessary types and functions.
#ifdef GTENSOR_DEVICE_CUDA
#include <cufft.h>
#endif

// ======================================================================
// error handling helper

#define gtFFTCheck(what)                                                       \
  {                                                                            \
    gtFFTCheckImpl(what, __FILE__, __LINE__);                                  \
  }

inline void gtFFTCheckImpl(cufftResult_t code, const char* file, int line)
{
  if (code != CUFFT_SUCCESS) {
    fprintf(stderr, "gtFFTCheck: status %d at %s:%d\n", code, file, line);
    abort();
  }
}

namespace gt
{

namespace fft
{

namespace detail
{

inline cufftResult_t cufftExecC2C_forward(cufftHandle plan, cufftComplex* idata,
                                          cufftComplex* odata)
{
  return cufftExecC2C(plan, idata, odata, CUFFT_FORWARD);
}

inline cufftResult_t cufftExecC2C_inverse(cufftHandle plan, cufftComplex* idata,
                                          cufftComplex* odata)
{
  return cufftExecC2C(plan, idata, odata, CUFFT_INVERSE);
}

inline cufftResult_t cufftExecZ2Z_forward(cufftHandle plan,
                                          cufftDoubleComplex* idata,
                                          cufftDoubleComplex* odata)
{
  return cufftExecZ2Z(plan, idata, odata, CUFFT_FORWARD);
}

inline cufftResult_t cufftExecZ2Z_inverse(cufftHandle plan,
                                          cufftDoubleComplex* idata,
                                          cufftDoubleComplex* odata)
{
  return cufftExecZ2Z(plan, idata, odata, CUFFT_INVERSE);
}

template <gt::fft::Domain D, typename R>
struct fft_config;

template <>
struct fft_config<gt::fft::Domain::COMPLEX, double>
{
  constexpr static cufftType type_forward = CUFFT_Z2Z;
  constexpr static cufftType type_inverse = CUFFT_Z2Z;
  constexpr static auto exec_fn_forward = &cufftExecZ2Z_forward;
  constexpr static auto exec_fn_inverse = &cufftExecZ2Z_inverse;
  using Tin = gt::complex<double>;
  using Tout = gt::complex<double>;
  using Bin = cufftDoubleComplex;
  using Bout = cufftDoubleComplex;
};

template <>
struct fft_config<gt::fft::Domain::COMPLEX, float>
{
  constexpr static cufftType type_forward = CUFFT_C2C;
  constexpr static cufftType type_inverse = CUFFT_C2C;
  constexpr static auto exec_fn_forward = &cufftExecC2C_forward;
  constexpr static auto exec_fn_inverse = &cufftExecC2C_inverse;
  using Tin = gt::complex<float>;
  using Tout = gt::complex<float>;
  using Bin = cufftComplex;
  using Bout = cufftComplex;
};

template <>
struct fft_config<gt::fft::Domain::REAL, double>
{
  constexpr static cufftType type_forward = CUFFT_D2Z;
  constexpr static cufftType type_inverse = CUFFT_Z2D;
  constexpr static auto exec_fn_forward = &cufftExecD2Z;
  constexpr static auto exec_fn_inverse = &cufftExecZ2D;
  using Tin = double;
  using Tout = gt::complex<double>;
  using Bin = cufftDoubleReal;
  using Bout = cufftDoubleComplex;
};

template <>
struct fft_config<gt::fft::Domain::REAL, float>
{
  constexpr static cufftType type_forward = CUFFT_R2C;
  constexpr static cufftType type_inverse = CUFFT_C2R;
  constexpr static auto exec_fn_forward = &cufftExecR2C;
  constexpr static auto exec_fn_inverse = &cufftExecC2R;
  using Tin = float;
  using Tout = gt::complex<float>;
  using Bin = cufftReal;
  using Bout = cufftComplex;
};

} // namespace detail

template <gt::fft::Domain D, typename R>
class FFTPlanManyCUDA
{
public:
  FFTPlanManyCUDA(std::vector<int> lengths, int batch_size = 1,
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
    init(lengths, 1, idist, 1, odist, batch_size, stream);
  }

  FFTPlanManyCUDA(std::vector<int> lengths, int istride, int idist, int ostride,
                  int odist, int batch_size = 1,
                  gt::stream_view stream = gt::stream_view{})
    : is_valid_(true)
  {
    init(lengths, istride, idist, ostride, odist, batch_size, stream);
  }

  // move only
  // delete copy ctor/assign
  FFTPlanManyCUDA(const FFTPlanManyCUDA& other) = delete;
  FFTPlanManyCUDA& operator=(const FFTPlanManyCUDA& other) = delete;

  // custom move to avoid double destroy in moved-from object
  FFTPlanManyCUDA(FFTPlanManyCUDA&& other) : is_valid_(true)
  {
    plan_forward_ = other.plan_forward_;
    plan_inverse_ = other.plan_inverse_;
    is_layout_asymmetric_ = other.is_layout_asymmetric_;
    other.is_valid_ = false;
  }

  FFTPlanManyCUDA& operator=(FFTPlanManyCUDA&& other)
  {
    plan_forward_ = other.plan_forward_;
    plan_inverse_ = other.plan_inverse_;
    is_layout_asymmetric_ = other.is_layout_asymmetric_;
    other.is_valid_ = false;
    return *this;
  }

  virtual ~FFTPlanManyCUDA()
  {
    if (is_valid_) {
      cufftDestroy(plan_forward_);
      if (is_layout_asymmetric_) {
        cufftDestroy(plan_inverse_);
      }
    }
  }

  void operator()(typename detail::fft_config<D, R>::Tin* indata,
                  typename detail::fft_config<D, R>::Tout* outdata) const
  {
    if (!is_valid_) {
      throw std::runtime_error("can't use a moved-from plan");
    }
    using Bin = typename detail::fft_config<D, R>::Bin;
    using Bout = typename detail::fft_config<D, R>::Bout;
    auto bin = reinterpret_cast<Bin*>(indata);
    auto bout = reinterpret_cast<Bout*>(outdata);
    auto fn = detail::fft_config<D, R>::exec_fn_forward;
    gtFFTCheck(fn(plan_forward_, bin, bout));
  }

  void inverse(typename detail::fft_config<D, R>::Tout* indata,
               typename detail::fft_config<D, R>::Tin* outdata) const
  {
    if (!is_valid_) {
      throw std::runtime_error("can't use a moved-from plan");
    }
    using Bin = typename detail::fft_config<D, R>::Bin;
    using Bout = typename detail::fft_config<D, R>::Bout;
    auto bin = reinterpret_cast<Bout*>(indata);
    auto bout = reinterpret_cast<Bin*>(outdata);
    auto fn = detail::fft_config<D, R>::exec_fn_inverse;
    if (D == gt::fft::Domain::REAL) {
      gtFFTCheck(fn(plan_inverse_, bin, bout));
    } else if (is_layout_asymmetric_) {
      gtFFTCheck(fn(plan_inverse_, bin, bout));
    } else {
      gtFFTCheck(fn(plan_forward_, bin, bout));
    }
  }

  std::size_t get_work_buffer_bytes() { return work_buffer_bytes_; }

private:
  void init(std::vector<int> lengths, int istride, int idist, int ostride,
            int odist, int batch_size, gt::stream_view stream)
  {
    int rank = lengths.size();
    int* n = lengths.data();

    work_buffer_bytes_ = 0;

    std::vector<int> freq_lengths = lengths;
    if (D == gt::fft::Domain::REAL) {
      freq_lengths[rank - 1] = lengths[rank - 1] / 2 + 1;
      is_layout_asymmetric_ = true;
    } else if (istride != ostride || idist != odist) {
      is_layout_asymmetric_ = true;
    } else {
      is_layout_asymmetric_ = false;
    }
    int* nfreq = freq_lengths.data();

    std::size_t plan_work_bytes = 0;

    auto cuda_stream = stream.get_backend_stream();
    auto type_forward = detail::fft_config<D, R>::type_forward;
    auto type_inverse = detail::fft_config<D, R>::type_inverse;
    gtFFTCheck(cufftPlanMany(&plan_forward_, rank, n, n, istride, idist, nfreq,
                             ostride, odist, type_forward, batch_size));
    gtFFTCheck(cufftSetStream(plan_forward_, cuda_stream));
    gtFFTCheck(cufftGetSize(plan_forward_, &plan_work_bytes));
    work_buffer_bytes_ += plan_work_bytes;
    if (is_layout_asymmetric_) {
      gtFFTCheck(cufftPlanMany(&plan_inverse_, rank, n, nfreq, ostride, odist,
                               n, istride, idist, type_inverse, batch_size));
      gtFFTCheck(cufftSetStream(plan_inverse_, cuda_stream));
      gtFFTCheck(cufftGetSize(plan_inverse_, &plan_work_bytes));
      work_buffer_bytes_ += plan_work_bytes;
    }
  }

  cufftHandle plan_forward_;
  cufftHandle plan_inverse_;
  bool is_valid_;
  // flag to keep track of whether an inverse plan is required - it is needed
  // when the input and output layout are different, which is always true for
  // REAL domain and sometimes for COMPLEX
  bool is_layout_asymmetric_;

  std::size_t work_buffer_bytes_;
};

template <gt::fft::Domain D, typename R>
using FFTPlanManyBackend = FFTPlanManyCUDA<D, R>;

} // namespace fft

} // namespace gt

#endif // GTENSOR_FFT_CUDA_H
