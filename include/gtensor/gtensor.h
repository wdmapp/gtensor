
#ifndef GTENSOR_GTENSOR_H
#define GTENSOR_GTENSOR_H

#include "defs.h"
#include "device_backend.h"

#include "complex.h"
#include "complex_ops.h"
#include "gcontainer.h"
#include "gfunction.h"
#include "gtensor_span.h"
#include "gview.h"
#include "operator.h"

namespace gt
{

// ======================================================================
// gtensor

// forward declared in gtensor_span.h
// template <typename T, size_type N, typename S>
// class gtensor;

template <typename T, size_type N, typename S>
struct gtensor_inner_types<gtensor<T, N, S>>
{
  using space_type = S;
  constexpr static size_type dimension = N;

  using storage_type =
    typename space::space_traits<S>::template storage_type<T>;
  using value_type = typename storage_type::value_type;
  using pointer = typename storage_type::pointer;
  using const_pointer = typename storage_type::const_pointer;
  using reference = typename storage_type::reference;
  using const_reference = typename storage_type::const_reference;
};

template <typename T, size_type N, typename S = space::host>
class gtensor : public gcontainer<gtensor<T, N, S>>
{
public:
  using self_type = gtensor<T, N, S>;
  using base_type = gcontainer<self_type>;
  using inner_types = gtensor_inner_types<self_type>;
  using storage_type = typename inner_types::storage_type;

  using value_type = typename inner_types::value_type;
  using const_reference = typename inner_types::const_reference;
  using reference = typename inner_types::reference;
  using const_pointer = typename inner_types::const_pointer;
  using pointer = typename inner_types::pointer;

  using typename base_type::shape_type;
  using typename base_type::strides_type;

  using kernel_type = gtensor_span<T, N, S>;
  using const_kernel_type = gtensor_span<std::add_const_t<T>, N, S>;

  using base_type::dimension;

  using base_type::base_type;
  gtensor() = default;
  explicit gtensor(const shape_type& shape);
  gtensor(helper::nd_initializer_list_t<T, N> il);
  template <typename E>
  gtensor(const expression<E>& e);
  template <typename E,
            typename = std::enable_if_t<std::is_convertible<E, T>::value>>
  gtensor(const shape_type& shape, E fill_value);

  using base_type::operator=;

  const_kernel_type to_kernel() const;
  kernel_type to_kernel();

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
// gtensor implementation

template <typename T, size_type N, typename S>
inline gtensor<T, N, S>::gtensor(const shape_type& shape)
  : base_type(shape, calc_strides(shape)), storage_(calc_size(shape))
{}

template <typename T, size_type N, typename S>
template <typename E, typename Enabled>
inline gtensor<T, N, S>::gtensor(const shape_type& shape, E fill_value)
  : base_type(shape, calc_strides(shape)), storage_(calc_size(shape))
{
  this->fill(fill_value);
}

template <typename T, size_type N, typename S>
inline gtensor<T, N, S>::gtensor(helper::nd_initializer_list_t<T, N> il)
  : base_type({}, {})
{
  // FIXME?! this kinda changes row-major list into transposed col-major array
  shape_type shape = helper::nd_initializer_list_shape<N>(il);
  base_type::resize(shape);
#if defined(GTENSOR_HAVE_DEVICE) && !defined(GTENSOR_USE_THRUST)
  if (std::is_same<S, space::device>::value) {
    gtensor<T, N, space::host> host_temp(shape);
    helper::nd_initializer_list_copy<N>(il, host_temp);
    gt::backend::system::copy<space::host, space::device>(
      host_temp.data(), base_type::data(), host_temp.size());
  } else {
    helper::nd_initializer_list_copy<N>(il, (*this));
  }
#else
  helper::nd_initializer_list_copy<N>(il, (*this));
#endif
}

template <typename T, size_type N, typename S>
template <typename E>
inline gtensor<T, N, S>::gtensor(const expression<E>& e)
{
  this->resize(e.derived().shape());
  *this = e.derived();
}

template <typename T, size_type N, typename S>
GT_INLINE auto gtensor<T, N, S>::storage_impl() const -> const storage_type&
{
  return storage_;
}

template <typename T, size_type N, typename S>
GT_INLINE auto gtensor<T, N, S>::storage_impl() -> storage_type&
{
  return storage_;
}

#pragma nv_exec_check_disable
template <typename T, size_type N, typename S>
GT_INLINE auto gtensor<T, N, S>::data_access_impl(size_t i) const
  -> const_reference
{
  return storage_[i];
}

#pragma nv_exec_check_disable
template <typename T, size_type N, typename S>
GT_INLINE auto gtensor<T, N, S>::data_access_impl(size_t i) -> reference
{
  return storage_[i];
}

template <typename T, size_type N, typename S>
inline auto gtensor<T, N, S>::to_kernel() const -> const_kernel_type
{
  return const_kernel_type(this->data(), this->shape(), this->strides());
}

template <typename T, size_type N, typename S>
inline auto gtensor<T, N, S>::to_kernel() -> kernel_type
{
  return kernel_type(this->data(), this->shape(), this->strides());
}

// ======================================================================
// copies
//
// FIXME, there should be only one, more general version,
// and maybe this should be .assign or operator=

template <typename T, size_type N, typename S_from, typename S_to>
void copy(const gtensor<T, N, S_from>& from, gtensor<T, N, S_to>& to)
{
  assert(from.size() == to.size());
  gt::backend::system::copy<S_from, S_to>(from.data(), to.data(), to.size());
}

template <typename T, size_type N, typename S_from, typename S_to>
void copy(const gtensor_span<T, N, S_from>& from, gtensor<T, N, S_to>& to)
{
  assert(from.size() == to.size());
  gt::backend::system::copy<S_from, S_to>(from.data(), to.data(), to.size());
}

template <typename T, size_type N, typename S_from, typename S_to>
void copy(const gtensor<T, N, S_from>& from, gtensor_span<T, N, S_to>& to)
{
  assert(from.size() == to.size());
  gt::backend::system::copy<S_from, S_to>(from.data(), to.data(), to.size());
}

template <typename T, size_type N, typename S_from, typename S_to>
void copy(const gtensor_span<T, N, S_from>& from, gtensor_span<T, N, S_to>& to)
{
  assert(from.size() == to.size());
  gt::backend::system::copy<S_from, S_to>(from.data(), to.data(), to.size());
}

// ======================================================================
// launch

#if defined(GTENSOR_DEVICE_CUDA) || defined(GTENSOR_DEVICE_HIP)

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

#endif // CUDA or HIP

namespace detail
{
template <int N, typename Sp>
struct launch;

template <>
struct launch<1, space::host>
{
  template <typename F>
  static void run(const gt::shape_type<1>& shape, F&& f)
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
  static void run(const gt::shape_type<2>& shape, F&& f)
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
  static void run(const gt::shape_type<3>& shape, F&& f)
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
  static void run(const gt::shape_type<4>& shape, F&& f)
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
  static void run(const gt::shape_type<5>& shape, F&& f)
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
  static void run(const gt::shape_type<6>& shape, F&& f)
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
template <>
struct launch<1, space::device>
{
  template <typename F>
  static void run(const gt::shape_type<1>& shape, F&& f)
  {
    const int BS_1D = 256;
    dim3 numThreads(BS_1D);
    dim3 numBlocks((shape[0] + BS_1D - 1) / BS_1D);

    gpuSyncIfEnabled();
    gtLaunchKernel(kernel_launch, numBlocks, numThreads, 0, 0, shape,
                   std::forward<F>(f));
    gpuSyncIfEnabled();
  }
};

template <>
struct launch<2, space::device>
{
  template <typename F>
  static void run(const gt::shape_type<2>& shape, F&& f)
  {
    dim3 numThreads(BS_X, BS_Y);
    dim3 numBlocks((shape[0] + BS_X - 1) / BS_X, (shape[1] + BS_Y - 1) / BS_Y);

    gpuSyncIfEnabled();
    gtLaunchKernel(kernel_launch, numBlocks, numThreads, 0, 0, shape,
                   std::forward<F>(f));
    gpuSyncIfEnabled();
  }
};

template <>
struct launch<3, space::device>
{
  template <typename F>
  static void run(const gt::shape_type<3>& shape, F&& f)
  {
    dim3 numThreads(BS_X, BS_Y);
    dim3 numBlocks((shape[0] + BS_X - 1) / BS_X, (shape[1] + BS_Y - 1) / BS_Y,
                   shape[2]);

    gpuSyncIfEnabled();
    gtLaunchKernel(kernel_launch, numBlocks, numThreads, 0, 0, shape,
                   std::forward<F>(f));
    gpuSyncIfEnabled();
  }
};

template <>
struct launch<4, space::device>
{
  template <typename F>
  static void run(const gt::shape_type<4>& shape, F&& f)
  {
    dim3 numThreads(BS_X, BS_Y);
    dim3 numBlocks((shape[0] + BS_X - 1) / BS_X, (shape[1] + BS_Y - 1) / BS_Y,
                   shape[2] * shape[3]);

    gpuSyncIfEnabled();
    gtLaunchKernel(kernel_launch, numBlocks, numThreads, 0, 0, shape,
                   std::forward<F>(f));
    gpuSyncIfEnabled();
  }
};

template <>
struct launch<5, space::device>
{
  template <typename F>
  static void run(const gt::shape_type<5>& shape, F&& f)
  {
    dim3 numThreads(BS_X, BS_Y);
    dim3 numBlocks((shape[0] + BS_X - 1) / BS_X, (shape[1] + BS_Y - 1) / BS_Y,
                   shape[2] * shape[3] * shape[4]);

    gtLaunchKernel(kernel_launch, numBlocks, numThreads, 0, 0, shape,
                   std::forward<F>(f));
  }
};

#elif defined(GTENSOR_DEVICE_SYCL)

template <>
struct launch<1, space::device>
{
  template <typename F>
  static void run(const gt::shape_type<1>& shape, F&& f)
  {
    sycl::queue& q = gt::backend::sycl::get_queue();
    auto range = sycl::range<1>(shape[0]);
    auto e = q.submit([&](sycl::handler& cgh) {
      using kname = gt::backend::sycl::Launch1<decltype(f)>;
      cgh.parallel_for<kname>(range, [=](sycl::item<1> item) {
        int i = item.get_id(0);
        f(i);
      });
    });
    e.wait();
  }
};

template <>
struct launch<2, space::device>
{
  template <typename F>
  static void run(const gt::shape_type<2>& shape, F&& f)
  {
    sycl::queue& q = gt::backend::sycl::get_queue();
    auto range = sycl::range<2>(shape[0], shape[1]);
    auto e = q.submit([&](sycl::handler& cgh) {
      using kname = gt::backend::sycl::Launch2<decltype(f)>;
      cgh.parallel_for<kname>(range, [=](sycl::item<2> item) {
        int i = item.get_id(0);
        int j = item.get_id(1);
        f(i, j);
      });
    });
    e.wait();
  }
};

template <>
struct launch<3, space::device>
{
  template <typename F>
  static void run(const gt::shape_type<3>& shape, F&& f)
  {
    sycl::queue& q = gt::backend::sycl::get_queue();
    auto range = sycl::range<3>(shape[0], shape[1], shape[2]);
    auto e = q.submit([&](sycl::handler& cgh) {
      using kname = gt::backend::sycl::Launch3<decltype(f)>;
      cgh.parallel_for<kname>(range, [=](sycl::item<3> item) {
        int i = item.get_id(0);
        int j = item.get_id(1);
        int k = item.get_id(2);
        f(i, j, k);
      });
    });
    e.wait();
  }
};

template <size_type N>
struct launch<N, space::device>
{
  template <typename F>
  static void run(const gt::shape_type<N>& shape, F&& f)
  {
    sycl::queue& q = gt::backend::sycl::get_queue();
    int size = calc_size(shape);
    // use linear indexing for simplicity
    auto block_size = std::min(size, BS_LINEAR);
    auto strides = calc_strides(shape);
    auto range =
      sycl::nd_range<1>(sycl::range<1>(size), sycl::range<1>(block_size));
    auto e = q.submit([&](sycl::handler& cgh) {
      using kname = gt::backend::sycl::LaunchN<decltype(f)>;
      cgh.parallel_for<kname>(range, [=](sycl::nd_item<1> item) {
        int i = item.get_global_id(0);
        auto idx = unravel(i, strides);
        index_expression(f, idx);
      });
    });
    e.wait();
  }
};

#endif

} // namespace detail

template <int N, typename F>
inline void launch_host(const gt::shape_type<N>& shape, F&& f)
{
  detail::launch<N, space::host>::run(shape, std::forward<F>(f));
}

template <int N, typename F>
inline void launch(const gt::shape_type<N>& shape, F&& f)
{
  detail::launch<N, space::device>::run(shape, std::forward<F>(f));
}

template <int N, typename S, typename F>
inline void launch(const gt::shape_type<N>& shape, F&& f)
{
  detail::launch<N, S>::run(shape, std::forward<F>(f));
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
inline std::enable_if_t<is_gcontainer<E>::value, E> eval(E&& e)
{
  return std::forward<E>(e);
}

template <typename E>
inline std::enable_if_t<
  !is_gcontainer<E>::value,
  gtensor<expr_value_type<E>, expr_dimension<E>(), expr_space_type<E>>>
eval(E&& e)
{
  return {std::forward<E>(e)};
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

} // namespace gt

#endif
