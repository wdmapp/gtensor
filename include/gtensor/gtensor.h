
#ifndef GTENSOR_GTENSOR_H
#define GTENSOR_GTENSOR_H

#include <sstream>
#include <string>

#include "defs.h"
#include "device_backend.h"

#include "complex.h"
#include "complex_ops.h"
#include "gcontainer.h"
#include "gfunction.h"
#include "gtensor_forward.h"
#include "gtensor_span.h"
#include "gview.h"
#include "helper.h"
#include "operator.h"
#include "space.h"

#if defined(GTENSOR_ENABLE_FP16)
#include "complex_float16_t.h"
#include "float16_t.h"
#endif

namespace gt
{

// ======================================================================
// gtensor_container

template <typename EC, size_type N>
struct gtensor_inner_types<gtensor_container<EC, N>>
{
  using space_type = typename space::storage_traits<EC>::space_type;
  constexpr static size_type dimension = N;

  using storage_type = EC;
  using value_type = typename storage_type::value_type;
  using pointer = typename storage_type::pointer;
  using const_pointer = typename storage_type::const_pointer;
  using reference = typename storage_type::reference;
  using const_reference = typename storage_type::const_reference;
};

template <typename EC, size_type N>
class gtensor_container : public gcontainer<gtensor_container<EC, N>>
{
public:
  using self_type = gtensor_container<EC, N>;
  using base_type = gcontainer<self_type>;
  using inner_types = gtensor_inner_types<self_type>;
  using storage_type = typename inner_types::storage_type;
  using space_type = typename base_type::space_type;

  using value_type = typename inner_types::value_type;
  using const_reference = typename inner_types::const_reference;
  using reference = typename inner_types::reference;
  using const_pointer = typename inner_types::const_pointer;
  using pointer = typename inner_types::pointer;

  using typename base_type::shape_type;
  using typename base_type::strides_type;

  using kernel_type = gtensor_span<value_type, N, space_type>;
  using const_kernel_type =
    gtensor_span<std::add_const_t<value_type>, N, space_type>;

  using base_type::dimension;

  using base_type::base_type;
  gtensor_container() = default;
  explicit gtensor_container(const shape_type& shape);
  gtensor_container(helper::nd_initializer_list_t<value_type, N> il);
  template <typename E>
  gtensor_container(const expression<E>& e);
  template <typename E, typename = std::enable_if_t<
                          std::is_convertible<E, value_type>::value>>
  gtensor_container(const shape_type& shape, E fill_value);

  using base_type::operator=;

  const_kernel_type to_kernel() const;
  kernel_type to_kernel();

  std::string typestr() const&;

  bool is_f_contiguous() const;

private:
  GT_INLINE const storage_type& storage_impl() const;
  GT_INLINE storage_type& storage_impl();
  GT_INLINE const_reference data_access_impl(size_type i) const;
  GT_INLINE reference data_access_impl(size_type i);

  storage_type storage_;

  friend class gstrided<self_type>;
  friend class gcontainer<self_type>;
};

// ======================================================================
// gtensor_container implementation

template <typename T, size_type N>
inline gtensor_container<T, N>::gtensor_container(const shape_type& shape)
  : base_type(shape, calc_strides(shape)), storage_(calc_size(shape))
{
#if defined(GTENSOR_USE_THRUST) && defined(GTENSOR_DEVICE_HIP)
  // NOTE: rocThrust backend appears to not synchronize uninitialized zero fill
  // by default, which can lead to suprising behavior where data is zeroed after
  // constructor is run.
  gt::synchronize();
#endif
}

template <typename T, size_type N>
template <typename E, typename Enabled>
inline gtensor_container<T, N>::gtensor_container(const shape_type& shape,
                                                  E fill_value)
  : base_type(shape, calc_strides(shape)), storage_(calc_size(shape))
{
  this->fill(fill_value);
}

template <typename EC, size_type N>
inline gtensor_container<EC, N>::gtensor_container(
  helper::nd_initializer_list_t<value_type, N> il)
  : base_type({}, {})
{
  // FIXME?! this kinda changes row-major list into transposed col-major array
  shape_type shape = helper::nd_initializer_list_shape<N>(il);
  base_type::resize(shape);
#if defined(GTENSOR_HAVE_DEVICE) && !defined(GTENSOR_USE_THRUST)
  if (std::is_same<space_type, space::device>::value) {
    gtensor<value_type, N, space::host> host_temp(shape);
    helper::nd_initializer_list_copy<N>(il, host_temp);
    gt::copy_n(host_temp.data(), host_temp.size(), base_type::data());
  } else {
    helper::nd_initializer_list_copy<N>(il, (*this));
  }
#else
  helper::nd_initializer_list_copy<N>(il, (*this));
#endif
}

template <typename T, size_type N>
template <typename E>
inline gtensor_container<T, N>::gtensor_container(const expression<E>& e)
{
  this->resize(e.derived().shape());
  *this = e.derived();
}

template <typename T, size_type N>
GT_INLINE auto gtensor_container<T, N>::storage_impl() const
  -> const storage_type&
{
  return storage_;
}

template <typename T, size_type N>
GT_INLINE auto gtensor_container<T, N>::storage_impl() -> storage_type&
{
  return storage_;
}

#pragma nv_exec_check_disable
template <typename T, size_type N>
GT_INLINE auto gtensor_container<T, N>::data_access_impl(size_t i) const
  -> const_reference
{
  return storage_[i];
}

#pragma nv_exec_check_disable
template <typename T, size_type N>
GT_INLINE auto gtensor_container<T, N>::data_access_impl(size_t i) -> reference
{
  return storage_[i];
}

template <typename T, size_type N>
inline auto gtensor_container<T, N>::to_kernel() const -> const_kernel_type
{
  return const_kernel_type(this->data(), this->shape(), this->strides());
}

template <typename T, size_type N>
inline auto gtensor_container<T, N>::to_kernel() -> kernel_type
{
  return kernel_type(this->data(), this->shape(), this->strides());
}

template <typename T, size_type N>
inline std::string gtensor_container<T, N>::typestr() const&
{
  std::stringstream s;
  s << "d" << N << "<" << get_type_name<typename T::value_type>() << ">"
    << this->shape() << this->strides();
  return s.str();
}

template <typename T, size_type N>
inline bool gtensor_container<T, N>::is_f_contiguous() const
{
  return true;
}

// ======================================================================
// launch

#if defined(GTENSOR_DEVICE_CUDA) || defined(GTENSOR_DEVICE_HIP)

#ifdef GTENSOR_PER_DIM_KERNELS

template <typename F>
__global__ void kernel_launch(gt::shape_type<1> shape, F f)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < shape[0])
    f(i);
}

template <typename F>
__global__ void kernel_launch(gt::shape_type<2> shape, F f)
{
  int i = threadIdx.x + blockIdx.x * BS_X;
  int j = threadIdx.y + blockIdx.y * BS_Y;

  if (i < shape[0] && j < shape[1]) {
    f(i, j);
  }
}

template <typename F>
__global__ void kernel_launch(gt::shape_type<3> shape, F f)
{
  int i = threadIdx.x + blockIdx.x * BS_X;
  int j = threadIdx.y + blockIdx.y * BS_Y;
  int k = blockIdx.z;

  if (i < shape[0] && j < shape[1]) {
    f(i, j, k);
  }
}

template <typename F>
__global__ void kernel_launch(gt::shape_type<4> shape, F f)
{
  int i = threadIdx.x + blockIdx.x * BS_X;
  int j = threadIdx.y + blockIdx.y * BS_Y;
  int b = blockIdx.z;
  int l = b / shape[2];
  b -= l * shape[2];
  int k = b;

  if (i < shape[0] && j < shape[1]) {
    f(i, j, k, l);
  }
}

template <typename F>
__global__ void kernel_launch(gt::shape_type<5> shape, F f)
{
  int i = threadIdx.x + blockIdx.x * BS_X;
  int j = threadIdx.y + blockIdx.y * BS_Y;
  int b = blockIdx.z;
  int m = b / (shape[2] * shape[3]);
  b -= m * (shape[2] * shape[3]);
  int l = b / shape[2];
  b -= l * shape[2];
  int k = b;

  if (i < shape[0] && j < shape[1]) {
    f(i, j, k, l, m);
  }
}

template <typename F>
__global__ void kernel_launch(gt::shape_type<6> shape, F f)
{
  int i = threadIdx.x + blockIdx.x * BS_X;
  int j = threadIdx.y + blockIdx.y * BS_Y;
  int b = blockIdx.z;
  int n = b / (shape[2] * shape[3] * shape[4]);
  b -= n * (shape[2] * shape[3] * shape[4]);
  int m = b / (shape[2] * shape[3]);
  b -= m * (shape[2] * shape[3]);
  int l = b / shape[2];
  b -= l * shape[2];
  int k = b;

  if (i < shape[0] && j < shape[1]) {
    f(i, j, k, l, m, n);
  }
}

#else // not GTENSOR_PER_DIM_KERNELS

template <typename F, size_type N>
__global__ void kernel_launch_N(F f, size_type size, gt::shape_type<N> strides)
{
  // workaround ROCm 5.2.0 compiler bug
  size_type i = threadIdx.x + static_cast<size_type>(blockIdx.x) * blockDim.x;

  if (i < size) {
    auto idx = unravel(i, strides);
    index_expression(f, idx);
  }
}

#endif // GTENSOR_PER_DIM_KERNELS

#endif // CUDA or HIP

namespace detail
{
template <int N, typename Sp>
struct launch;

template <>
struct launch<1, space::host>
{
  template <typename F>
  static void run(const gt::shape_type<1>& shape, F&& f, gt::stream_view stream)
  {
    for (int i = 0; i < shape[0]; i++) {
      std::forward<F>(f)(i);
    }
  }
};

template <>
struct launch<2, space::host>
{
  template <typename F>
  static void run(const gt::shape_type<2>& shape, F&& f, gt::stream_view stream)
  {
    for (int j = 0; j < shape[1]; j++) {
      for (int i = 0; i < shape[0]; i++) {
        std::forward<F>(f)(i, j);
      }
    }
  }
};

template <>
struct launch<3, space::host>
{
  template <typename F>
  static void run(const gt::shape_type<3>& shape, F&& f, gt::stream_view stream)
  {
    for (int k = 0; k < shape[2]; k++) {
      for (int j = 0; j < shape[1]; j++) {
        for (int i = 0; i < shape[0]; i++) {
          std::forward<F>(f)(i, j, k);
        }
      }
    }
  }
};

template <>
struct launch<4, space::host>
{
  template <typename F>
  static void run(const gt::shape_type<4>& shape, F&& f, gt::stream_view stream)
  {
    for (int l = 0; l < shape[3]; l++) {
      for (int k = 0; k < shape[2]; k++) {
        for (int j = 0; j < shape[1]; j++) {
          for (int i = 0; i < shape[0]; i++) {
            std::forward<F>(f)(i, j, k, l);
          }
        }
      }
    }
  }
};

template <>
struct launch<5, space::host>
{
  template <typename F>
  static void run(const gt::shape_type<5>& shape, F&& f, gt::stream_view stream)
  {
    for (int m = 0; m < shape[4]; m++) {
      for (int l = 0; l < shape[3]; l++) {
        for (int k = 0; k < shape[2]; k++) {
          for (int j = 0; j < shape[1]; j++) {
            for (int i = 0; i < shape[0]; i++) {
              std::forward<F>(f)(i, j, k, l, m);
            }
          }
        }
      }
    }
  }
};

template <>
struct launch<6, space::host>
{
  template <typename F>
  static void run(const gt::shape_type<6>& shape, F&& f, gt::stream_view stream)
  {
    for (int n = 0; n < shape[5]; n++) {
      for (int m = 0; m < shape[4]; m++) {
        for (int l = 0; l < shape[3]; l++) {
          for (int k = 0; k < shape[2]; k++) {
            for (int j = 0; j < shape[1]; j++) {
              for (int i = 0; i < shape[0]; i++) {
                std::forward<F>(f)(i, j, k, l, m, n);
              }
            }
          }
        }
      }
    }
  }
};

#if defined(GTENSOR_DEVICE_CUDA) || defined(GTENSOR_DEVICE_HIP)

#ifdef GTENSOR_PER_DIM_KERNELS

template <>
struct launch<1, space::device>
{
  template <typename F>
  static void run(const gt::shape_type<1>& shape, F&& f, gt::stream_view stream)
  {
    const int BS_1D = 256;
    dim3 numThreads(BS_1D);
    dim3 numBlocks((shape[0] + BS_1D - 1) / BS_1D);

    gpuSyncIfEnabledStream(stream);
    gtLaunchKernel(kernel_launch, numBlocks, numThreads, 0,
                   stream.get_backend_stream(), shape, std::forward<F>(f));
    gpuSyncIfEnabledStream(stream);
  }
};

template <>
struct launch<2, space::device>
{
  template <typename F>
  static void run(const gt::shape_type<2>& shape, F&& f, gt::stream_view stream)
  {
    dim3 numThreads(BS_X, BS_Y);
    dim3 numBlocks((shape[0] + BS_X - 1) / BS_X, (shape[1] + BS_Y - 1) / BS_Y);

    gpuSyncIfEnabledStream(stream);
    gtLaunchKernel(kernel_launch, numBlocks, numThreads, 0,
                   stream.get_backend_stream(), shape, std::forward<F>(f));
    gpuSyncIfEnabledStream(stream);
  }
};

template <>
struct launch<3, space::device>
{
  template <typename F>
  static void run(const gt::shape_type<3>& shape, F&& f, gt::stream_view stream)
  {
    dim3 numThreads(BS_X, BS_Y);
    dim3 numBlocks((shape[0] + BS_X - 1) / BS_X, (shape[1] + BS_Y - 1) / BS_Y,
                   shape[2]);

    gpuSyncIfEnabledStream(stream);
    gtLaunchKernel(kernel_launch, numBlocks, numThreads, 0,
                   stream.get_backend_stream(), shape, std::forward<F>(f));
    gpuSyncIfEnabledStream(stream);
  }
};

template <>
struct launch<4, space::device>
{
  template <typename F>
  static void run(const gt::shape_type<4>& shape, F&& f, gt::stream_view stream)
  {
    dim3 numThreads(BS_X, BS_Y);
    dim3 numBlocks((shape[0] + BS_X - 1) / BS_X, (shape[1] + BS_Y - 1) / BS_Y,
                   shape[2] * shape[3]);

    gpuSyncIfEnabledStream(stream);
    gtLaunchKernel(kernel_launch, numBlocks, numThreads, 0,
                   stream.get_backend_stream(), shape, std::forward<F>(f));
    gpuSyncIfEnabledStream(stream);
  }
};

template <>
struct launch<5, space::device>
{
  template <typename F>
  static void run(const gt::shape_type<5>& shape, F&& f, gt::stream_view stream)
  {
    dim3 numThreads(BS_X, BS_Y);
    dim3 numBlocks((shape[0] + BS_X - 1) / BS_X, (shape[1] + BS_Y - 1) / BS_Y,
                   shape[2] * shape[3] * shape[4]);

    gtLaunchKernel(kernel_launch, numBlocks, numThreads, 0,
                   stream.get_backend_stream(), shape, std::forward<F>(f));
  }
};

template <>
struct launch<6, space::device>
{
  template <typename F>
  static void run(const gt::shape_type<6>& shape, F&& f, gt::stream_view stream)
  {
    dim3 numThreads(BS_X, BS_Y);
    dim3 numBlocks((shape[0] + BS_X - 1) / BS_X, (shape[1] + BS_Y - 1) / BS_Y,
                   shape[2] * shape[3] * shape[4] * shape[5]);

    gtLaunchKernel(kernel_launch, numBlocks, numThreads, 0,
                   stream.get_backend_stream(), shape, std::forward<F>(f));
  }
};

#else // not GTENSOR_PER_DIM_KERNELS

template <int N>
struct launch<N, space::device>
{
  template <typename F>
  static void run(const gt::shape_type<N>& shape, F&& f, gt::stream_view stream)
  {
    auto size = calc_size(shape);
    auto strides = calc_strides(shape);
    unsigned int block_size = BS_LINEAR;
    if (block_size > size) {
      block_size = static_cast<unsigned int>(size);
    }

    dim3 numThreads(block_size);
    dim3 numBlocks(gt::div_ceil(size, block_size));

    gpuSyncIfEnabledStream(stream);
    gtLaunchKernel(kernel_launch_N, numBlocks, numThreads, 0,
                   stream.get_backend_stream(), std::forward<F>(f), size,
                   strides);
    gpuSyncIfEnabledStream(stream);
  }
};

#endif // GTENSOR_PER_DIM_KERNELS

#elif defined(GTENSOR_DEVICE_SYCL)

#ifdef GTENSOR_PER_DIM_KERNELS

template <>
struct launch<1, space::device>
{
  template <typename F>
  static void run(const gt::shape_type<1>& shape, F&& f, gt::stream_view stream)
  {
    gpuSyncIfEnabledStream(stream);

    sycl::queue q = stream.get_backend_stream();
    auto range = sycl::range<1>(shape[0]);
    auto e = q.submit([&](sycl::handler& cgh) {
      // using kname = gt::backend::sycl::Launch1<decltype(f)>;
      cgh.parallel_for(range, [=](sycl::item<1> item) {
        auto i = item.get_id(0);
        f(i);
      });
    });

    gpuSyncIfEnabledStream(stream);
  }
};

template <>
struct launch<2, space::device>
{
  template <typename F>
  static void run(const gt::shape_type<2>& shape, F&& f, gt::stream_view stream)
  {
    gpuSyncIfEnabledStream(stream);

    sycl::queue q = stream.get_backend_stream();
    auto range = sycl::range<2>(shape[1], shape[0]);
    auto e = q.submit([&](sycl::handler& cgh) {
      // using kname = gt::backend::sycl::Launch2<decltype(f)>;
      cgh.parallel_for(range, [=](sycl::item<2> item) {
        auto i = item.get_id(1);
        auto j = item.get_id(0);
        f(i, j);
      });
    });

    gpuSyncIfEnabledStream(stream);
  }
};

template <>
struct launch<3, space::device>
{
  template <typename F>
  static void run(const gt::shape_type<3>& shape, F&& f, gt::stream_view stream)
  {
    gpuSyncIfEnabledStream(stream);

    sycl::queue q = stream.get_backend_stream();
    auto range = sycl::range<3>(shape[2], shape[1], shape[0]);
    auto e = q.submit([&](sycl::handler& cgh) {
      // using kname = gt::backend::sycl::Launch3<decltype(f)>;
      cgh.parallel_for(range, [=](sycl::item<3> item) {
        auto i = item.get_id(2);
        auto j = item.get_id(1);
        auto k = item.get_id(0);
        f(i, j, k);
      });
    });

    gpuSyncIfEnabledStream(stream);
  }
};

#endif

template <size_type N>
struct launch<N, space::device>
{
  template <typename F>
  static void run(const gt::shape_type<N>& shape, F&& f, gt::stream_view stream)
  {
    gpuSyncIfEnabledStream(stream);

    sycl::queue q = stream.get_backend_stream();
    auto size = calc_size(shape);
    auto strides = calc_strides(shape);
    auto e = q.submit([&](sycl::handler& cgh) {
      // using kname = gt::backend::sycl::LaunchN<decltype(f)>;
      cgh.parallel_for(sycl::range<1>(size), [=](sycl::id<1> i) {
        auto idx = unravel(i, strides);
        index_expression(f, idx);
      });
    });

    gpuSyncIfEnabledStream(stream);
  }
};

#endif

} // namespace detail

template <int N, typename F>
inline void launch_host(const gt::shape_type<N>& shape, F&& f,
                        gt::stream_view stream = gt::stream_view{})
{
  detail::launch<N, space::host>::run(shape, std::forward<F>(f), stream);
}

template <int N, typename F>
inline void launch(const gt::shape_type<N>& shape, F&& f,
                   gt::stream_view stream = gt::stream_view{})
{
  detail::launch<N, space::device>::run(shape, std::forward<F>(f), stream);
}

template <int N, typename S, typename F>
inline void launch(const gt::shape_type<N>& shape, F&& f,
                   gt::stream_view stream = gt::stream_view{})
{
  detail::launch<N, S>::run(shape, std::forward<F>(f), stream);
}

// ======================================================================
// gtensor_device, gtensor_span_device

template <typename T, size_type N>
using gtensor_device = gtensor<T, N, space::device>;

template <typename T, size_type N>
using gtensor_span_device = gtensor_span<T, N, space::device>;

// ======================================================================
// empty

template <typename T, typename S = gt::space::host, size_type N>
inline auto empty(const gt::shape_type<N> shape)
{
  return gtensor<T, N, S>(shape);
}

template <typename T, typename S = gt::space::host, size_type N>
inline auto empty(const int (&shape)[N])
{
  return gtensor<T, N, S>(gt::shape_type<N>(shape));
}

template <typename T, size_type N>
inline auto empty_device(const gt::shape_type<N> shape)
{
  return gtensor<T, N, gt::space::device>(shape);
}

template <typename T, size_type N>
inline auto empty_device(const int (&shape)[N])
{
  return gtensor<T, N, gt::space::device>(gt::shape_type<N>(shape));
}

// ======================================================================
// full

template <typename T, typename S = gt::space::host, size_type N>
inline auto full(const gt::shape_type<N> shape, T fill_value)
{
  return gtensor<T, N, S>(shape, fill_value);
}

template <typename T, typename S = gt::space::host, size_type N>
inline auto full(const int (&shape)[N], T fill_value)
{
  return gtensor<T, N, S>(gt::shape_type<N>(shape), fill_value);
}

template <typename T, size_type N>
inline auto full_device(const gt::shape_type<N> shape, T fill_value)
{
  return gtensor<T, N, gt::space::device>(shape, fill_value);
}

template <typename T, size_type N>
inline auto full_device(const int (&shape)[N], T fill_value)
{
  return gtensor<T, N, gt::space::device>(gt::shape_type<N>(shape), fill_value);
}

// ======================================================================
// zeros

template <typename T, typename S = gt::space::host, size_type N>
inline auto zeros(const gt::shape_type<N> shape)
{
  return gtensor<T, N, S>(shape, 0);
}

template <typename T, typename S = gt::space::host, size_type N>
inline auto zeros(const int (&shape)[N])
{
  return gtensor<T, N, S>(gt::shape_type<N>(shape), 0);
}

template <typename T, size_type N>
inline auto zeros_device(const gt::shape_type<N> shape)
{
  return gtensor<T, N, gt::space::device>(shape, 0);
}

template <typename T, size_type N>
inline auto zeros_device(const int (&shape)[N])
{
  return gtensor<T, N, gt::space::device>(gt::shape_type<N>(shape), 0);
}

// ======================================================================
// empty_like

template <typename E>
inline auto empty_like(const expression<E>& _e)
{
  const auto& e = _e.derived();
  return gtensor<expr_value_type<E>, expr_dimension<E>(), expr_space_type<E>>(
    e.shape());
}

// ======================================================================
// full_like

template <typename E, typename T,
          typename =
            std::enable_if_t<std::is_convertible<T, expr_value_type<E>>::value>>
inline auto full_like(const expression<E>& _e, T v)
{
  const auto& e = _e.derived();
  return gtensor<expr_value_type<E>, expr_dimension<E>(), expr_space_type<E>>(
    e.shape(), v);
}

// ======================================================================
// zeros_like

template <typename E>
inline auto zeros_like(const expression<E>& _e)
{
  const auto& e = _e.derived();
  return gtensor<expr_value_type<E>, expr_dimension<E>(), expr_space_type<E>>(
    e.shape(), 0);
}

// ======================================================================
// eval

template <typename E>
inline std::enable_if_t<
  is_gcontainer<E>::value && !std::is_const<expr_value_type<E>>::value, E>
eval(E&& e)
{
  return std::forward<E>(e);
}

template <typename E>
inline std::enable_if_t<
  is_gcontainer<E>::value && !std::is_const<expr_value_type<E>>::value, E&>
eval(E& e)
{
  return e;
}

template <typename E>
inline std::enable_if_t<!is_gcontainer<E>::value ||
                          std::is_const<expr_value_type<E>>::value,
                        gtensor<std::remove_cv_t<expr_value_type<E>>,
                                expr_dimension<E>(), expr_space_type<E>>>
eval(E&& e)
{
  return {std::forward<E>(e)};
}

// ======================================================================
// has_data_and_size

template <typename E, typename Enable = void>
struct has_data_and_size : std::false_type
{};

template <typename E>
struct has_data_and_size<E,
                         gt::meta::void_t<decltype(std::declval<E>().data()),
                                          decltype(std::declval<E>().size())>>
  : std::true_type
{};

// ======================================================================
// copies

template <typename SRC, typename DST>
std::enable_if_t<gt::has_data_and_size<SRC>::value &&
                 gt::has_data_and_size<DST>::value>
copy(const SRC& src, DST&& dst)
{
  if (!dst.is_f_contiguous()) {
    auto dst_tmp = gt::empty_like(dst);
    gt::copy(src, dst_tmp);
    dst = dst_tmp;
  } else {
    if (src.is_f_contiguous()) {
      assert(src.size() == dst.size());
      gt::copy_n(src.data(), src.size(), dst.data());
    } else {
      gt::copy(gt::eval(src), dst);
    }
  }
}

// if both expressions are in the same space, we can just assign
template <typename SRC, typename DST>
std::enable_if_t<
  std::is_same<expr_space_type<SRC>, expr_space_type<DST>>::value &&
  !(gt::has_data_and_size<SRC>::value && gt::has_data_and_size<DST>::value)>
copy(const SRC& src, DST&& dst)
{
  dst = src;
}

// different spaces, source not storage like, destination is storage-like
template <typename SRC, typename DST>
std::enable_if_t<
  !std::is_same<expr_space_type<SRC>, expr_space_type<DST>>::value &&
  (!gt::has_data_and_size<SRC>::value && gt::has_data_and_size<DST>::value)>
copy(const SRC& src, DST&& dst)
{
  gt::copy(gt::eval(src), dst);
}

// different spaces, destination is not storage-like
template <typename SRC, typename DST>
std::enable_if_t<
  !std::is_same<expr_space_type<SRC>, expr_space_type<DST>>::value &&
  !gt::has_data_and_size<DST>::value>
copy(const SRC& src, DST&& dst)
{
  auto dst_tmp = gt::empty_like(dst);
  gt::copy(src, dst_tmp);
  dst = dst_tmp;
}

// ======================================================================
// arange

namespace detail
{

template <typename T>
class arange_generator_1d
{
public:
  arange_generator_1d(T start, T step) : start_(start), step_(step) {}

  GT_INLINE T operator()(int i) const { return start_ + T(i) * step_; }

private:
  T start_;
  T step_;
};

} // namespace detail

template <typename T>
inline auto arange(T start, T end, T step = 1)
{
  auto shape = gt::shape((end - start) / step);
  return generator<1, T>(shape, detail::arange_generator_1d<T>(start, step));
}

// ======================================================================
// host_mirror

namespace detail
{

template <typename E, typename Enable = void>
struct host_mirror
{
  static auto run(const E& e)
  {
    // FIXME, empty_like with space would be helpful
    return gt::empty<gt::expr_value_type<E>>(e.shape());
  }
};

// specialization if the expression is already on the host: just return a
// reference to it
template <typename E>
struct host_mirror<E, std::enable_if_t<std::is_same<gt::expr_space_type<E>,
                                                    gt::space::host>::value>>
{
  static E& run(E& e) { return e; }
};

} // namespace detail

template <typename E>
decltype(auto) host_mirror(E& e)
{
  return detail::host_mirror<E>::run(e);
}

} // namespace gt

#endif
