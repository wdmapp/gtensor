
#ifndef GTENSOR_BACKEND_SYCL_H
#define GTENSOR_BACKEND_SYCL_H

#include <cstdlib>
#include <exception>
#include <iostream>
#include <unordered_map>

#include <CL/sycl.hpp>

#ifdef GTENSOR_DEVICE_SYCL_L0
#include "level_zero/ze_api.h"
#include "level_zero/zes_api.h"

#include "CL/sycl/backend/level_zero.hpp"
#endif

#ifdef GTENSOR_DEVICE_SYCL_OPENCL
#include "CL/sycl/backend/opencl.hpp"
#endif

#include "backend_common.h"

#include "gtensor/backend_sycl_device.h"

// ======================================================================
// gt::backend::sycl

namespace gt
{

namespace backend
{

namespace sycl
{

// kernel name templates
template <typename E1, typename E2, typename K1, typename K2>
class Assign1;
template <typename E1, typename E2, typename K1, typename K2>
class Assign2;
template <typename E1, typename E2, typename K1, typename K2>
class Assign3;
template <typename E1, typename E2, typename K1, typename K2>
class AssignN;

template <typename F>
class Launch1;
template <typename F>
class Launch2;
template <typename F>
class Launch3;
template <typename F>
class LaunchN;

template <typename E>
class Sum;
template <typename E>
class Max;
template <typename E>
class Min;

} // namespace sycl

namespace allocator_impl
{
template <>
struct gallocator<gt::space::sycl>
{
  template <typename T>
  static T* allocate(size_type n)
  {
    return cl::sycl::malloc_device<T>(n, gt::backend::sycl::get_queue());
  }

  template <typename T>
  static void deallocate(T* p)
  {
    cl::sycl::free(p, gt::backend::sycl::get_queue());
  }
};

template <>
struct gallocator<gt::space::sycl_managed>
{
  template <typename T>
  static T* allocate(size_t n)
  {
    auto mtype = gt::backend::get_managed_memory_type();
    if (mtype == gt::backend::managed_memory_type::managed) {
      return cl::sycl::malloc_shared<T>(n, gt::backend::sycl::get_queue());
    } else if (mtype == gt::backend::managed_memory_type::device) {
      return cl::sycl::malloc_device<T>(n, gt::backend::sycl::get_queue());
    } else {
      throw std::runtime_error("unsupported managed memory type for backend");
    }
  }

  template <typename T>
  static void deallocate(T* p)
  {
    cl::sycl::free(p, gt::backend::sycl::get_queue());
  }
};

template <>
struct gallocator<gt::space::sycl_host>
{
  template <typename T>
  static T* allocate(size_t n)
  {
    return cl::sycl::malloc_host<T>(n, gt::backend::sycl::get_queue());
  }

  template <typename T>
  static void deallocate(T* p)
  {
    cl::sycl::free(p, gt::backend::sycl::get_queue());
  }
};

} // namespace allocator_impl

namespace copy_impl
{

template <typename InputPtr, typename OutputPtr>
inline void sycl_copy_n(InputPtr in, size_type count, OutputPtr out)
{
  cl::sycl::queue& q = gt::backend::sycl::get_queue();
  auto in_raw = gt::raw_pointer_cast(in);
  auto out_raw = gt::raw_pointer_cast(out);

  auto e = q.copy(in_raw, out_raw, count);

  // sync if in/out is host/managed to mimic CUDA sync behavior, see
  // https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior
  // TODO: use async everywhere and require developer to sync after copy when
  // needed
  auto in_alloc_type = ::sycl::get_pointer_type(in_raw, q.get_context());
  auto out_alloc_type = ::sycl::get_pointer_type(out_raw, q.get_context());
  if (in_alloc_type != ::sycl::usm::alloc::device ||
      out_alloc_type != ::sycl::usm::alloc::device) {
    e.wait();
  }
}

template <typename InputPtr, typename OutputPtr>
inline void copy_n(gt::space::sycl tag_in, gt::space::sycl tag_out, InputPtr in,
                   size_type count, OutputPtr out)
{
  sycl_copy_n(in, count, out);
}

template <typename InputPtr, typename OutputPtr>
inline void copy_n(gt::space::sycl tag_in, gt::space::host tag_out, InputPtr in,
                   size_type count, OutputPtr out)
{
  sycl_copy_n(in, count, out);
}

template <typename InputPtr, typename OutputPtr>
inline void copy_n(gt::space::host tag_in, gt::space::sycl tag_out, InputPtr in,
                   size_type count, OutputPtr out)
{
  sycl_copy_n(in, count, out);
}

#if 0
template <typename InputPtr, typename OutputPtr>
inline void copy_n(gt::space::host tag_in, gt::space::host tag_out, InputPtr in,
                   size_type count, OutputPtr out)
{
  sycl_copy_n(in, count, out);
}
#endif

} // namespace copy_impl

namespace fill_impl
{
template <typename Ptr, typename T>
inline void fill(gt::space::sycl tag, Ptr first, Ptr last, const T& value)
{
  using element_type = typename gt::pointer_traits<Ptr>::element_type;
  cl::sycl::queue& q = gt::backend::sycl::get_queue();
  cl::sycl::event e;
  auto first_raw = gt::raw_pointer_cast(first);
  if (element_type(value) == element_type()) {
    e = q.memset(first_raw, 0, (last - first) * sizeof(element_type));
  } else {
    assert(sizeof(element_type) == 1);
    e = q.memset(first_raw, value, (last - first) * sizeof(element_type));
  }

  // sync if pointer is host/managed to mimic CUDA sync behavior, see
  // https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior
  // TODO: use async everywhere and require developer to sync after copy when
  // needed
  auto alloc_type = ::sycl::get_pointer_type(first_raw, q.get_context());
  if (alloc_type != ::sycl::usm::alloc::device) {
    e.wait();
  }
} // namespace fill_impl

} // namespace fill_impl

template <>
class backend_ops<gt::space::sycl>
{
public:
  static void device_synchronize() { gt::backend::sycl::get_queue().wait(); }

  static int device_get_count()
  {
    return gt::backend::sycl::device::get_sycl_queues_instance()
      .get_device_count();
  }

  static void device_set(int device_id)
  {
    gt::backend::sycl::device::get_sycl_queues_instance().set_device_id(
      device_id);
  }

  static int device_get()
  {
    return gt::backend::sycl::device::get_sycl_queues_instance()
      .get_device_id();
  }

  static uint32_t device_get_vendor_id(int device_id)
  {
    return gt::backend::sycl::device::get_sycl_queues_instance()
      .get_device_vendor_id(device_id);
  }

  template <typename Ptr>
  static bool is_device_accessible(const Ptr ptr)
  {
    auto& q = gt::backend::sycl::get_queue();
    auto alloc_type = ::sycl::get_pointer_type(ptr, q.get_context());
    return (alloc_type == ::sycl::usm::alloc::device ||
            alloc_type == ::sycl::usm::alloc::shared);
  }

  template <typename Ptr>
  static memory_type get_memory_type(const Ptr ptr)
  {
    auto& q = gt::backend::sycl::get_queue();
    auto alloc_type = ::sycl::get_pointer_type(ptr, q.get_context());
    switch (alloc_type) {
      case ::sycl::usm::alloc::host: return memory_type::host;
      case ::sycl::usm::alloc::device: return memory_type::device;
      case ::sycl::usm::alloc::shared: return memory_type::managed;
      case ::sycl::usm::alloc::unknown: return memory_type::unregistered;
      default:
        fprintf(stderr, "ERROR: unknown memoryType %d.\n", alloc_type);
        std::abort();
    }
  }

  template <typename T>
  static void prefetch_device(T* p, size_type n)
  {
    auto& q = gt::backend::sycl::get_queue();
    q.prefetch(p, n);
  }

  // Not available in SYCL 2020, make it a no-op
  template <typename T>
  static void prefetch_host(T* p, size_type n)
  {}

  template <typename T>
  static void copy_async_dd(const T* src, T* dst, size_type count)
  {
    auto& q = gt::backend::sycl::get_queue();
    q.memcpy(dst, src, sizeof(T) * count);
  }

  class stream_view
    : public stream_interface::stream_view_base<cl::sycl::queue&>
  {
  public:
    using base_class = stream_view_base<cl::sycl::queue&>;
    using base_class::base_class;

    bool is_default()
    {
      return this->stream_ == gt::backend::sycl::get_queue();
    }

    void synchronize() { stream_.wait(); }
  };
};

namespace stream_interface
{

using sycl_stream_t = cl::sycl::queue&;

template <>
inline sycl_stream_t create<sycl_stream_t>()
{
  return gt::backend::sycl::new_stream_queue();
}

template <>
inline sycl_stream_t get_default<sycl_stream_t>()
{
  return gt::backend::sycl::get_queue();
}

template <>
inline void destroy<sycl_stream_t>(sycl_stream_t q)
{
  return gt::backend::sycl::delete_stream_queue(q);
}

} // namespace stream_interface

} // namespace backend

using stream_view = backend::backend_ops<gt::space::sycl>::stream_view;
using stream =
  backend::stream_interface::stream_base<cl::sycl::queue&, stream_view>;

} // namespace gt

#endif // GTENSOR_BACKEND_SYCL_H
