#ifndef GTENSOR_FFT_HIP_H
#define GTENSOR_FFT_HIP_H

#include <numeric>
#include <stdexcept>
#include <vector>

#include <rocfft/rocfft.h>

// ======================================================================
// error handling helper

#define gtFFTCheck(what)                                                       \
  {                                                                            \
    gtFFTCheckImpl(what, __FILE__, __LINE__);                                  \
  }

inline void gtFFTCheckImpl(rocfft_status code, const char* file, int line)
{
  if (code != rocfft_status_success) {
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

template <gt::fft::Domain D, typename R>
struct fft_config;

template <>
struct fft_config<gt::fft::Domain::COMPLEX, double>
{
  constexpr static rocfft_transform_type type_forward =
    rocfft_transform_type_complex_forward;
  constexpr static rocfft_transform_type type_inverse =
    rocfft_transform_type_complex_inverse;
  constexpr static rocfft_precision precision = rocfft_precision_double;
  constexpr static rocfft_array_type in_array_type =
    rocfft_array_type_complex_interleaved;
  constexpr static rocfft_array_type out_array_type =
    rocfft_array_type_complex_interleaved;
  using Tin = gt::complex<double>;
  using Tout = gt::complex<double>;
};

template <>
struct fft_config<gt::fft::Domain::COMPLEX, float>
{
  constexpr static rocfft_transform_type type_forward =
    rocfft_transform_type_complex_forward;
  constexpr static rocfft_transform_type type_inverse =
    rocfft_transform_type_complex_inverse;
  constexpr static rocfft_precision precision = rocfft_precision_single;
  constexpr static rocfft_array_type in_array_type =
    rocfft_array_type_complex_interleaved;
  constexpr static rocfft_array_type out_array_type =
    rocfft_array_type_complex_interleaved;
  using Tin = gt::complex<float>;
  using Tout = gt::complex<float>;
};

template <>
struct fft_config<gt::fft::Domain::REAL, double>
{
  constexpr static rocfft_transform_type type_forward =
    rocfft_transform_type_real_forward;
  constexpr static rocfft_transform_type type_inverse =
    rocfft_transform_type_real_inverse;
  constexpr static rocfft_precision precision = rocfft_precision_double;
  constexpr static rocfft_array_type in_array_type = rocfft_array_type_real;
  constexpr static rocfft_array_type out_array_type =
    rocfft_array_type_hermitian_interleaved;
  using Tin = double;
  using Tout = gt::complex<double>;
};

template <>
struct fft_config<gt::fft::Domain::REAL, float>
{
  constexpr static rocfft_transform_type type_forward =
    rocfft_transform_type_real_forward;
  constexpr static rocfft_transform_type type_inverse =
    rocfft_transform_type_real_inverse;
  constexpr static rocfft_precision precision = rocfft_precision_single;
  constexpr static rocfft_array_type in_array_type = rocfft_array_type_real;
  constexpr static rocfft_array_type out_array_type =
    rocfft_array_type_hermitian_interleaved;
  using Tin = float;
  using Tout = gt::complex<float>;
};

struct rocfft_initializer
{
  rocfft_initializer() { rocfft_setup(); }
  ~rocfft_initializer() { rocfft_cleanup(); }
};

} // namespace detail

template <gt::fft::Domain D, typename R>
class FFTPlanManyHIP
{
public:
  FFTPlanManyHIP(std::vector<int> lengths, int batch_size = 1,
                 gt::stream_view stream = gt::stream_view{})
    : is_valid_(true), work_buffer_(nullptr)
  {
    init(lengths, 1, 0, 1, 0, batch_size, stream);
  }

  FFTPlanManyHIP(std::vector<int> lengths, int istride, int idist, int ostride,
                 int odist, int batch_size = 1,
                 gt::stream_view stream = gt::stream_view{})
    : is_valid_(true), work_buffer_(nullptr)
  {
    init(lengths, istride, idist, ostride, odist, batch_size, stream);
  }

  // move only
  // delete copy ctor/assign
  FFTPlanManyHIP(const FFTPlanManyHIP& other) = delete;
  FFTPlanManyHIP& operator=(const FFTPlanManyHIP& other) = delete;

  // custom move to avoid double destroy in moved-from object
  FFTPlanManyHIP(FFTPlanManyHIP&& other) : is_valid_(true)
  {
    plan_forward_ = other.plan_forward_;
    plan_inverse_ = other.plan_inverse_;
    info_forward_ = other.info_forward_;
    info_inverse_ = other.info_inverse_;
    work_buffer_ = other.work_buffer_;
    work_buffer_bytes_ = other.work_buffer_bytes_;
    is_valid_ = true;
    other.is_valid_ = false;
  }

  FFTPlanManyHIP& operator=(FFTPlanManyHIP&& other)
  {
    plan_forward_ = other.plan_forward_;
    plan_inverse_ = other.plan_inverse_;
    info_forward_ = other.info_forward_;
    info_inverse_ = other.info_inverse_;
    work_buffer_ = other.work_buffer_;
    work_buffer_bytes_ = other.work_buffer_bytes_;
    is_valid_ = true;
    other.is_valid_ = false;
    return *this;
  }

  virtual ~FFTPlanManyHIP()
  {
    if (is_valid_) {
      rocfft_plan_destroy(plan_forward_);
      rocfft_plan_destroy(plan_inverse_);
      rocfft_execution_info_destroy(info_forward_);
      rocfft_execution_info_destroy(info_inverse_);
      if (work_buffer_ != nullptr) {
        static_cast<void>(hipFree(work_buffer_));
      }
    }
  }

  void operator()(typename detail::fft_config<D, R>::Tin* indata,
                  typename detail::fft_config<D, R>::Tout* outdata) const
  {
    if (!is_valid_) {
      throw std::runtime_error("can't use a moved-from plan");
    }
    void* in[1] = {(void*)indata};
    void* out[1] = {(void*)outdata};
    gtFFTCheck(rocfft_execute(plan_forward_, in, out, info_forward_));
  }

  void inverse(typename detail::fft_config<D, R>::Tout* indata,
               typename detail::fft_config<D, R>::Tin* outdata) const
  {
    if (!is_valid_) {
      throw std::runtime_error("can't use a moved-from plan");
    }
    void* in[1] = {(void*)indata};
    void* out[1] = {(void*)outdata};
    gtFFTCheck(rocfft_execute(plan_inverse_, in, out, info_inverse_));
  }

  std::size_t get_work_buffer_bytes() { return work_buffer_bytes_; }

private:
  void init(std::vector<int> lengths_int, int istride, int idist, int ostride,
            int odist, int batch_size, gt::stream_view stream)
  {
    int rank = lengths_int.size();

    // reverse lengths and convert to size_t
    std::vector<std::size_t> lengths(rank);
    for (int i = 0; i < rank; i++) {
      lengths[i] = lengths_int[rank - 1 - i];
    }
    std::size_t* n = lengths.data();

    // handle call to rocfft_setup/destroy
    static detail::rocfft_initializer init;

    std::vector<std::size_t> freq_lengths = lengths;
    if (D == gt::fft::Domain::REAL) {
      freq_lengths[0] = lengths[0] / 2 + 1;
    }

    size_t istrides[3] = {1, 1, 1};
    size_t ostrides[3] = {1, 1, 1};
    size_t idist_st = idist;
    size_t odist_st = odist;

    rocfft_plan_description desc_forward;
    rocfft_plan_description desc_inverse;

    if (idist_st == 0) {
      idist_st = std::accumulate(lengths.begin(), lengths.end(), 1,
                                 std::multiplies<std::size_t>());
    }
    if (odist_st == 0) {
      odist_st = std::accumulate(freq_lengths.begin(), freq_lengths.end(), 1,
                                 std::multiplies<std::size_t>());
    }

    istrides[0] = istride;
    ostrides[0] = ostride;
    for (int i = 1; i < rank; i++) {
      istrides[i] = lengths[i - 1] * istrides[i - 1];
      ostrides[i] = freq_lengths[i - 1] * ostrides[i - 1];
    }

    rocfft_plan_description_create(&desc_forward);
    rocfft_plan_description_set_data_layout(
      desc_forward, detail::fft_config<D, R>::in_array_type,
      detail::fft_config<D, R>::out_array_type, 0, 0, rank, istrides, idist_st,
      rank, ostrides, odist_st);
    rocfft_plan_description_create(&desc_inverse);
    rocfft_plan_description_set_data_layout(
      desc_inverse, detail::fft_config<D, R>::out_array_type,
      detail::fft_config<D, R>::in_array_type, 0, 0, rank, ostrides, odist_st,
      rank, istrides, idist_st);

    size_t work_size_forward;
    size_t work_size_inverse;

    auto hip_stream = stream.get_backend_stream();
    auto type_forward = detail::fft_config<D, R>::type_forward;
    auto type_inverse = detail::fft_config<D, R>::type_inverse;
    auto precision = detail::fft_config<D, R>::precision;

    gtFFTCheck(rocfft_execution_info_create(&info_forward_));
    gtFFTCheck(rocfft_execution_info_set_stream(info_forward_, hip_stream));
    gtFFTCheck(rocfft_plan_create(&plan_forward_, rocfft_placement_notinplace,
                                  type_forward, precision, rank, n, batch_size,
                                  desc_forward));
    gtFFTCheck(
      rocfft_plan_get_work_buffer_size(plan_forward_, &work_size_forward));

    gtFFTCheck(rocfft_execution_info_create(&info_inverse_));
    gtFFTCheck(rocfft_execution_info_set_stream(info_inverse_, hip_stream));
    gtFFTCheck(rocfft_plan_create(&plan_inverse_, rocfft_placement_notinplace,
                                  type_inverse, precision, rank, n, batch_size,
                                  desc_inverse));
    gtFFTCheck(
      rocfft_plan_get_work_buffer_size(plan_inverse_, &work_size_inverse));

    work_buffer_bytes_ = std::max(work_size_forward, work_size_inverse);

    if (work_buffer_bytes_ > 0) {
      // use same work buffer for forward and inverse plans
      hipError_t malloc_result = hipMalloc(&work_buffer_, work_buffer_bytes_);
      if (malloc_result != hipSuccess) {
        fprintf(stderr, "gpuCheck: %d (%s) %s %d\n", malloc_result,
                hipGetErrorString(malloc_result), __FILE__, __LINE__);
        abort();
      }

      if (work_size_forward > 0) {
        gtFFTCheck(rocfft_execution_info_set_work_buffer(
          info_forward_, work_buffer_, work_size_forward));
      }
      if (work_size_inverse > 0) {
        gtFFTCheck(rocfft_execution_info_set_work_buffer(
          info_inverse_, work_buffer_, work_size_inverse));
      }
    }

    rocfft_plan_description_destroy(desc_forward);
    rocfft_plan_description_destroy(desc_inverse);
  }

  rocfft_plan plan_forward_;
  rocfft_plan plan_inverse_;
  rocfft_execution_info info_forward_;
  rocfft_execution_info info_inverse_;
  void* work_buffer_;
  size_t work_buffer_bytes_;

  bool is_valid_;
};

template <gt::fft::Domain D, typename R>
using FFTPlanManyBackend = FFTPlanManyHIP<D, R>;

} // namespace fft

} // namespace gt

#endif // GTENSOR_FFT_HIP_H
