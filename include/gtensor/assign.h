
#ifndef GTENSOR_ASSIGN_H
#define GTENSOR_ASSIGN_H

#include "defs.h"

namespace gt
{

constexpr const int BS_X = 16;
constexpr const int BS_Y = 16;
constexpr const int BS_LINEAR = 256;

// ======================================================================
// assign

namespace detail
{

template <size_type N, typename SP>
struct assigner
{
  static_assert(!std::is_same<SP, SP>::value, "assigner not implemented.");
};

template <>
struct assigner<1, space::host>
{
  template <typename E1, typename E2>
  static void run(E1& lhs, const E2& rhs, stream_view stream)
  {
    // printf("assigner<1, host>\n");
    for (int i = 0; i < lhs.shape(0); i++) {
      lhs(i) = rhs(i);
    }
  }
};

template <>
struct assigner<2, space::host>
{
  template <typename E1, typename E2>
  static void run(E1& lhs, const E2& rhs, stream_view stream)
  {
    // printf("assigner<2, host>\n");
    for (int j = 0; j < lhs.shape(1); j++) {
      for (int i = 0; i < lhs.shape(0); i++) {
        lhs(i, j) = rhs(i, j);
      }
    }
  }
};

template <>
struct assigner<3, space::host>
{
  template <typename E1, typename E2>
  static void run(E1& lhs, const E2& rhs, stream_view stream)
  {
    // printf("assigner<3, host>\n");
    for (int k = 0; k < lhs.shape(2); k++) {
      for (int j = 0; j < lhs.shape(1); j++) {
        for (int i = 0; i < lhs.shape(0); i++) {
          lhs(i, j, k) = rhs(i, j, k);
        }
      }
    }
  }
};

template <>
struct assigner<4, space::host>
{
  template <typename E1, typename E2>
  static void run(E1& lhs, const E2& rhs, stream_view stream)
  {
    // printf("assigner<4, host>\n");
    for (int l = 0; l < lhs.shape(3); l++) {
      for (int k = 0; k < lhs.shape(2); k++) {
        for (int j = 0; j < lhs.shape(1); j++) {
          for (int i = 0; i < lhs.shape(0); i++) {
            lhs(i, j, k, l) = rhs(i, j, k, l);
          }
        }
      }
    }
  }
};

template <>
struct assigner<5, space::host>
{
  template <typename E1, typename E2>
  static void run(E1& lhs, const E2& rhs, stream_view stream)
  {
    // printf("assigner<5, host>\n");
    for (int m = 0; m < lhs.shape(4); m++) {
      for (int l = 0; l < lhs.shape(3); l++) {
        for (int k = 0; k < lhs.shape(2); k++) {
          for (int j = 0; j < lhs.shape(1); j++) {
            for (int i = 0; i < lhs.shape(0); i++) {
              lhs(i, j, k, l, m) = rhs(i, j, k, l, m);
            }
          }
        }
      }
    }
  }
};

template <>
struct assigner<6, space::host>
{
  template <typename E1, typename E2>
  static void run(E1& lhs, const E2& rhs, stream_view stream)
  {
    // printf("assigner<6, host>\n");
    for (int n = 0; n < lhs.shape(5); n++) {
      for (int m = 0; m < lhs.shape(4); m++) {
        for (int l = 0; l < lhs.shape(3); l++) {
          for (int k = 0; k < lhs.shape(2); k++) {
            for (int j = 0; j < lhs.shape(1); j++) {
              for (int i = 0; i < lhs.shape(0); i++) {
                lhs(i, j, k, l, m, n) = rhs(i, j, k, l, m, n);
              }
            }
          }
        }
      }
    }
  }
};

#if defined(GTENSOR_DEVICE_CUDA) || defined(GTENSOR_DEVICE_HIP)

template <typename Elhs, typename Erhs>
__global__ void kernel_assign_1(Elhs lhs, Erhs rhs)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < lhs.shape(0)) {
    lhs(i) = rhs(i);
  }
}

template <typename Elhs, typename Erhs>
__global__ void kernel_assign_2(Elhs lhs, Erhs rhs)
{
  int i = threadIdx.x + blockIdx.x * BS_X;
  int j = threadIdx.y + blockIdx.y * BS_Y;

  if (i < lhs.shape(0) && j < lhs.shape(1)) {
    lhs(i, j) = rhs(i, j);
  }
}

template <typename Elhs, typename Erhs>
__global__ void kernel_assign_3(Elhs lhs, Erhs rhs)
{
  int i = threadIdx.x + blockIdx.x * BS_X;
  int j = threadIdx.y + blockIdx.y * BS_Y;
  int b = blockIdx.z;

  if (i < lhs.shape(0) && j < lhs.shape(1)) {
    lhs(i, j, b) = rhs(i, j, b);
  }
}

template <typename Elhs, typename Erhs>
__global__ void kernel_assign_4(Elhs lhs, Erhs rhs)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < lhs.size()) {
    int rem = idx;
    int i = rem % lhs.shape(0);
    rem /= lhs.shape(0);
    int j = rem % lhs.shape(1);
    rem /= lhs.shape(1);
    int k = rem % lhs.shape(2);
    rem /= lhs.shape(2);
    int l = rem;

    lhs(i, j, k, l) = rhs(i, j, k, l);
  }
}

template <typename Elhs, typename Erhs>
__global__ void kernel_assign_5(Elhs lhs, Erhs rhs)
{
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  int tidy = threadIdx.y + blockIdx.y * blockDim.y;
  int tidz = blockIdx.z;
  if (tidx < lhs.shape(0) * lhs.shape(1) &&
      tidy < lhs.shape(2) * lhs.shape(3)) {
    int j = tidx / lhs.shape(0), i = tidx % lhs.shape(0);
    int l = tidy / lhs.shape(2), k = tidy % lhs.shape(2);
    int m = tidz;

    lhs(i, j, k, l, m) = rhs(i, j, k, l, m);
  }
}

template <typename Elhs, typename Erhs>
__global__ void kernel_assign_6(Elhs lhs, Erhs _rhs)
{
  auto rhs = _rhs;
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  int tidy = threadIdx.y + blockIdx.y * blockDim.y;
  int tidz = blockIdx.z;
  if (tidx < lhs.shape(0) * lhs.shape(1) &&
      tidy < lhs.shape(2) * lhs.shape(3)) {
    int j = tidx / lhs.shape(0), i = tidx % lhs.shape(0);
    int l = tidy / lhs.shape(2), k = tidy % lhs.shape(2);
    int n = tidz / lhs.shape(4), m = tidz % lhs.shape(4);

    lhs(i, j, k, l, m, n) = rhs(i, j, k, l, m, n);
  }
}

template <typename Elhs, typename Erhs, size_type N>
__global__ void kernel_assign_N(Elhs lhs, Erhs rhs, int size,
                                gt::shape_type<N> strides)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < size) {
    auto idx = unravel(i, strides);
    index_expression(lhs, idx) = index_expression(rhs, idx);
  }
}

#ifdef GTENSOR_PER_DIM_KERNELS

template <>
struct assigner<1, space::device>
{
  template <typename E1, typename E2>
  static void run(E1& lhs, const E2& rhs, stream_view stream)
  {
    // printf("assigner<1, device>\n");
    dim3 numThreads(BS_LINEAR);
    dim3 numBlocks(gt::div_ceil(lhs.shape(0), BS_LINEAR));

    gpuSyncIfEnabled();
    gtLaunchKernel(kernel_assign_1, numBlocks, numThreads, 0,
                   stream.get_backend_stream(), lhs.to_kernel(),
                   rhs.to_kernel());
    gpuSyncIfEnabled();
  }
};

template <>
struct assigner<2, space::device>
{
  template <typename E1, typename E2>
  static void run(E1& lhs, const E2& rhs, stream_view stream)
  {
    // printf("assigner<2, device>\n");
    dim3 numThreads(BS_X, BS_Y);
    dim3 numBlocks((lhs.shape(0) + BS_X - 1) / BS_X,
                   (lhs.shape(1) + BS_Y - 1) / BS_Y);

    gpuSyncIfEnabled();
    gtLaunchKernel(kernel_assign_2, numBlocks, numThreads, 0,
                   stream.get_backend_stream(), lhs.to_kernel(),
                   rhs.to_kernel());
    gpuSyncIfEnabled();
  }
};

template <>
struct assigner<3, space::device>
{
  template <typename E1, typename E2>
  static void run(E1& lhs, const E2& rhs, stream_view stream)
  {
    // printf("assigner<3, device>\n");
    dim3 numThreads(BS_X, BS_Y);
    dim3 numBlocks((lhs.shape(0) + BS_X - 1) / BS_X,
                   (lhs.shape(1) + BS_Y - 1) / BS_Y, lhs.shape(2));

    gpuSyncIfEnabled();
    /*std::cout << "rhs " << typeid(rhs.to_kernel()).name() << "\n";
    std::cout << "numBlocks="<<numBlocks.x<<" "<<numBlocks.y<<" "<<numBlocks.z<<
    ", numThreads="<<numThreads.x<<" "<<numThreads.y<<" "<<numThreads.z<<"\n";*/
    gtLaunchKernel(kernel_assign_3, numBlocks, numThreads, 0,
                   stream.get_backend_stream(), lhs.to_kernel(),
                   rhs.to_kernel());
    gpuSyncIfEnabled();
  }
};

template <>
struct assigner<4, space::device>
{
  template <typename E1, typename E2>
  static void run(E1& lhs, const E2& rhs, stream_view stream)
  {
    // printf("assigner<4, device>\n");
    dim3 numThreads(256);
    dim3 numBlocks((lhs.size() + numThreads.x - 1) / numThreads.x);

    gpuSyncIfEnabled();
    // std::cout << "rhs " << typeid(rhs.to_kernel()).name() << "\n";
    gtLaunchKernel(kernel_assign_4, numBlocks, numThreads, 0,
                   stream.get_backend_stream(), lhs.to_kernel(),
                   rhs.to_kernel());
    gpuSyncIfEnabled();
  }
};

template <>
struct assigner<5, space::device>
{
  template <typename E1, typename E2>
  static void run(E1& lhs, const E2& rhs, stream_view stream)
  {
    // printf("assigner<6, device>\n");
    dim3 numThreads(BS_X, BS_Y);
    dim3 numBlocks((lhs.shape(0) * lhs.shape(1) + BS_X - 1) / BS_X,
                   (lhs.shape(2) * lhs.shape(3) + BS_Y - 1) / BS_Y,
                   lhs.shape(4));

    gpuSyncIfEnabled();
    // std::cout << "rhs " << typeid(rhs.to_kernel()).name() << "\n";
    gtLaunchKernel(kernel_assign_5, numBlocks, numThreads, 0,
                   stream.get_backend_stream(), lhs.to_kernel(),
                   rhs.to_kernel());
    gpuSyncIfEnabled();
  }
};

template <>
struct assigner<6, space::device>
{
  template <typename E1, typename E2>
  static void run(E1& lhs, const E2& rhs, stream_view stream)
  {
    // printf("assigner<6, device>\n");
    dim3 numThreads(BS_X, BS_Y);
    dim3 numBlocks((lhs.shape(0) * lhs.shape(1) + BS_X - 1) / BS_X,
                   (lhs.shape(2) * lhs.shape(3) + BS_Y - 1) / BS_Y,
                   lhs.shape(4) * lhs.shape(5));

    gpuSyncIfEnabled();
    // std::cout << "rhs " << typeid(rhs.to_kernel()).name() << "\n";
    gtLaunchKernel(kernel_assign_6, numBlocks, numThreads, 0,
                   stream.get_backend_stream(), lhs.to_kernel(),
                   rhs.to_kernel());
    gpuSyncIfEnabled();
  }
};

#endif

template <size_type N>
struct assigner<N, space::device>
{
  template <typename E1, typename E2>
  static void run(E1& lhs, const E2& rhs, stream_view stream)
  {
    int size = int(calc_size(lhs.shape()));
    auto strides = calc_strides(lhs.shape());
    auto block_size = std::min(size, BS_LINEAR);

    dim3 numThreads(block_size);
    dim3 numBlocks(gt::div_ceil(size, block_size));

    gpuSyncIfEnabled();
    gtLaunchKernel(kernel_assign_N, numBlocks, numThreads, 0,
                   stream.get_backend_stream(), lhs.to_kernel(),
                   rhs.to_kernel(), size, strides);
    gpuSyncIfEnabled();
  }
};

#elif defined(GTENSOR_DEVICE_SYCL)

#ifdef GTENSOR_PER_DIM_KERNELS

template <>
struct assigner<1, space::device>
{
  template <typename E1, typename E2>
  static void run(E1& lhs, const E2& rhs, gt::stream_view stream)
  {
    sycl::queue q = stream.get_backend_stream();
    auto k_lhs = lhs.to_kernel();
    auto k_rhs = rhs.to_kernel();
    auto range = sycl::range<1>(lhs.shape(0));
    auto e = q.submit([&](sycl::handler& cgh) {
      using ltype = decltype(k_lhs);
      using rtype = decltype(k_rhs);
      using kname = gt::backend::sycl::Assign1<E1, E2, ltype, rtype>;
      cgh.parallel_for<kname>(range, [=](sycl::item<1> item) {
        int i = item.get_id();
        k_lhs(i) = k_rhs(i);
      });
    });
    e.wait();
  }
};

template <>
struct assigner<2, space::device>
{
  template <typename E1, typename E2>
  static void run(E1& lhs, const E2& rhs, gt::stream_view stream)
  {
    sycl::queue q = stream.get_backend_stream();
    auto k_lhs = lhs.to_kernel();
    auto k_rhs = rhs.to_kernel();
    auto range = sycl::range<2>(lhs.shape(1), lhs.shape(0));
    auto e = q.submit([&](sycl::handler& cgh) {
      using ltype = decltype(k_lhs);
      using rtype = decltype(k_rhs);
      using kname = gt::backend::sycl::Assign2<E1, E2, ltype, rtype>;
      cgh.parallel_for<kname>(range, [=](sycl::item<2> item) {
        int i = item.get_id(1);
        int j = item.get_id(0);
        k_lhs(i, j) = k_rhs(i, j);
      });
    });
    e.wait();
  }
};

template <>
struct assigner<3, space::device>
{
  template <typename E1, typename E2>
  static void run(E1& lhs, const E2& rhs, gt::stream_view stream)
  {
    sycl::queue q = stream.get_backend_stream();
    auto k_lhs = lhs.to_kernel();
    auto k_rhs = rhs.to_kernel();
    auto range = sycl::range<3>(lhs.shape(2), lhs.shape(1), lhs.shape(0));
    auto e = q.submit([&](sycl::handler& cgh) {
      using ltype = decltype(k_lhs);
      using rtype = decltype(k_rhs);
      using kname = gt::backend::sycl::Assign3<E1, E2, ltype, rtype>;
      cgh.parallel_for<kname>(range, [=](sycl::item<3> item) {
        int i = item.get_id(2);
        int j = item.get_id(1);
        int k = item.get_id(0);
        k_lhs(i, j, k) = k_rhs(i, j, k);
      });
    });
    e.wait();
  }
};

#endif

template <size_type N>
struct assigner<N, space::device>
{
  template <typename E1, typename E2>
  static void run(E1& lhs, const E2& rhs, gt::stream_view stream)
  {
    sycl::queue q = stream.get_backend_stream();
    // use linear indexing for simplicity
    auto size = calc_size(lhs.shape());
    auto strides = calc_strides(lhs.shape());
    auto k_lhs = lhs.to_kernel();
    auto k_rhs = rhs.to_kernel();
    using ltype = decltype(k_lhs);
    using rtype = decltype(k_rhs);

    // Note: handle RHS that may be greater than 2k parameter limit
    gt::backend::device_storage<rtype> d_rhs(1);
    decltype(auto) d_rhs_p = gt::raw_pointer_cast(d_rhs.data());
    q.copy(&k_rhs, d_rhs_p, 1).wait();

    auto e = q.submit([&](sycl::handler& cgh) {
      using kname = gt::backend::sycl::AssignN<E1, E2, ltype, rtype>;
      cgh.parallel_for<kname>(sycl::range<1>(size), [=](sycl::id<1> i) {
        auto idx = unravel(i, strides);
        index_expression(k_lhs, idx) = index_expression(*d_rhs_p, idx);
      });
    });
    e.wait();
  }
};

#endif

} // namespace detail

template <typename E1, typename E2>
void assign(E1& lhs, const E2& rhs, gt::stream_view stream = gt::stream_view())
{
  static_assert(expr_dimension<E1>() == expr_dimension<E2>(),
                "cannot assign expressions of different dimension");
  // FIXME, need to check for brodcasting
#if 0
  if (lhs.shape() != rhs.shape()) {
    std::cout << "not the same shape! " << lhs.shape() << rhs.shape() << "\n";
  }
  assert(lhs.shape() == rhs.shape());
#endif
  detail::assigner<
    expr_dimension<E1>(),
    space_t<expr_space_type<E1>, expr_space_type<E2>>>::run(lhs, rhs, stream);
}

template <typename E1, typename T>
void assign(E1& lhs, const gscalar<T>& val,
            gt::stream_view stream = gt::stream_view())
{
  // FIXME, make more efficient
  detail::assigner<
    expr_dimension<E1>(),
    space_t<expr_space_type<E1>, expr_space_type<gscalar<T>>>>::run(lhs, val,
                                                                    stream);
}

} // namespace gt

#endif
