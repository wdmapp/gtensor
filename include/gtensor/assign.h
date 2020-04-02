
#ifndef GTENSOR_ASSIGN_H
#define GTENSOR_ASSIGN_H

#include "defs.h"

namespace gt
{

constexpr const int BS_X = 16;
constexpr const int BS_Y = 16;

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
  static void run(E1& lhs, const E2& rhs)
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
  static void run(E1& lhs, const E2& rhs)
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
  static void run(E1& lhs, const E2& rhs)
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
struct assigner<6, space::host>
{
  template <typename E1, typename E2>
  static void run(E1& lhs, const E2& rhs)
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

#ifdef GTENSOR_HAVE_DEVICE

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
  int i = threadIdx.x + blockIdx.x * BS_X;
  int j = threadIdx.y + blockIdx.y * BS_Y;
  int b = blockIdx.z;
  int l = b / lhs.shape(2);
  b -= l * lhs.shape(2);
  int k = b;

  if (i < lhs.shape(0) && j < lhs.shape(1)) {
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

template <>
struct assigner<1, space::device>
{
  template <typename E1, typename E2>
  static void run(E1& lhs, const E2& rhs)
  {
    // printf("assigner<1, device>\n");
    const int BS_1D = 256;
    dim3 numThreads(BS_1D);
    dim3 numBlocks((lhs.shape(0) + BS_1D - 1) / BS_1D);

    cudaSyncIfEnabled();
    gtLaunchKernel(kernel_assign_1, numBlocks, numThreads, 0, 0,
                   lhs.to_kernel(), rhs.to_kernel());
    cudaSyncIfEnabled();
  }
};

template <>
struct assigner<2, space::device>
{
  template <typename E1, typename E2>
  static void run(E1& lhs, const E2& rhs)
  {
    // printf("assigner<2, device>\n");
    dim3 numThreads(BS_X, BS_Y);
    dim3 numBlocks((lhs.shape(0) + BS_X - 1) / BS_X,
                   (lhs.shape(1) + BS_Y - 1) / BS_Y);

    cudaSyncIfEnabled();
    gtLaunchKernel(kernel_assign_2, numBlocks, numThreads, 0, 0,
                   lhs.to_kernel(), rhs.to_kernel());
    cudaSyncIfEnabled();
  }
};

template <>
struct assigner<3, space::device>
{
  template <typename E1, typename E2>
  static void run(E1& lhs, const E2& rhs)
  {
    // printf("assigner<3, device>\n");
    dim3 numThreads(BS_X, BS_Y);
    dim3 numBlocks((lhs.shape(0) + BS_X - 1) / BS_X,
                   (lhs.shape(1) + BS_Y - 1) / BS_Y, lhs.shape(2));

    cudaSyncIfEnabled();
    /*std::cout << "rhs " << typeid(rhs.to_kernel()).name() << "\n";
    std::cout << "numBlocks="<<numBlocks.x<<" "<<numBlocks.y<<" "<<numBlocks.z<<
    ", numThreads="<<numThreads.x<<" "<<numThreads.y<<" "<<numThreads.z<<"\n";*/
    gtLaunchKernel(kernel_assign_3, numBlocks, numThreads, 0, 0,
                   lhs.to_kernel(), rhs.to_kernel());
    cudaSyncIfEnabled();
  }
};

template <>
struct assigner<4, space::device>
{
  template <typename E1, typename E2>
  static void run(E1& lhs, const E2& rhs)
  {
    // printf("assigner<4, device>\n");
    dim3 numThreads(BS_X, BS_Y);
    dim3 numBlocks((lhs.shape(0) + BS_X - 1) / BS_X,
                   (lhs.shape(1) + BS_Y - 1) / BS_Y,
                   lhs.shape(2) * lhs.shape(3));

    cudaSyncIfEnabled();
    // std::cout << "rhs " << typeid(rhs.to_kernel()).name() << "\n";
    gtLaunchKernel(kernel_assign_4, numBlocks, numThreads, 0, 0,
                   lhs.to_kernel(), rhs.to_kernel());
    cudaSyncIfEnabled();
  }
};

template <>
struct assigner<5, space::device>
{
  template <typename E1, typename E2>
  static void run(E1& lhs, const E2& rhs)
  {
    // printf("assigner<6, device>\n");
    dim3 numThreads(BS_X, BS_Y);
    dim3 numBlocks((lhs.shape(0) * lhs.shape(1) + BS_X - 1) / BS_X,
                   (lhs.shape(2) * lhs.shape(3) + BS_Y - 1) / BS_Y,
                   lhs.shape(4));

    cudaSyncIfEnabled();
    // std::cout << "rhs " << typeid(rhs.to_kernel()).name() << "\n";
    gtLaunchKernel(kernel_assign_5, numBlocks, numThreads, 0, 0,
                   lhs.to_kernel(), rhs.to_kernel());
    cudaSyncIfEnabled();
  }
};

template <>
struct assigner<6, space::device>
{
  template <typename E1, typename E2>
  static void run(E1& lhs, const E2& rhs)
  {
    // printf("assigner<6, device>\n");
    dim3 numThreads(BS_X, BS_Y);
    dim3 numBlocks((lhs.shape(0) * lhs.shape(1) + BS_X - 1) / BS_X,
                   (lhs.shape(2) * lhs.shape(3) + BS_Y - 1) / BS_Y,
                   lhs.shape(4) * lhs.shape(5));

    cudaSyncIfEnabled();
    // std::cout << "rhs " << typeid(rhs.to_kernel()).name() << "\n";
    gtLaunchKernel(kernel_assign_6, numBlocks, numThreads, 0, 0,
                   lhs.to_kernel(), rhs.to_kernel());
    cudaSyncIfEnabled();
  }
};

#endif

} // namespace detail

template <typename E1, typename E2>
void assign(E1& lhs, const E2& rhs)
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
  detail::assigner<expr_dimension<E1>(),
                   space_t<expr_space_type<E1>, expr_space_type<E2>>>::run(lhs,
                                                                           rhs);
}

template <typename E1, typename T>
void assign(E1& lhs, const gscalar<T>& val)
{
  // FIXME, make more efficient
  detail::assigner<
    expr_dimension<E1>(),
    space_t<expr_space_type<E1>, expr_space_type<gscalar<T>>>>::run(lhs, val);
}

} // namespace gt

#endif
