
#ifndef GTENSOR_BACKEND_SYCL_H
#define GTENSOR_BACKEND_SYCL_H

#include "pointer_traits.h"
#include "sycl_backend.h"

#include <CL/sycl.hpp>

// ======================================================================
// gt::backend::sycl

namespace gt
{
namespace backend
{

namespace allocator_impl
{
template <>
struct gallocator<gt::space::sycl>
{
  template <typename T>
  static T* allocate(size_type n)
  {
    return cl::sycl::malloc_shared<T>(n, gt::backend::sycl::get_queue());
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
    return cl::sycl::malloc_shared<T>(n, gt::backend::sycl::get_queue());
  }

  template <typename T>
  static void deallocate(T* p)
  {
    cl::sycl::free(p, gt::backend::sycl::get_queue());
  }
}; // namespace allocator_impl

// The host allocation type in SYCL allows device code to directly access
// the data. This is generally not necessary or effecient for gtensor, so
// we opt for the same implementation as for the HOST device below.

template <>
struct gallocator<gt::space::sycl_host>

{
  template <typename T>
  static T* allocate(size_type n)
  {
    T* p = static_cast<T*>(malloc(sizeof(T) * n));
    if (!p) {
      std::cerr << "host allocate failed" << std::endl;
      std::abort();
    }
    return p;
  }

  template <typename T>
  static void deallocate(T* p)
  {
    free(p);
  }
};

// template <typename T>
// struct host
// {
//   static T* allocate( : size_type count)
//   {
//     return cl::sycl::malloc_host<T>(count, gt::backend::sycl::get_queue());
//   }

//   static void deallocate(T* p)
//   {
//     cl::sycl::free(p, gt::backend::sycl::get_queue());
//   }
// };

} // namespace allocator_impl

namespace sycl
{

inline void device_synchronize()
{
  gt::backend::sycl::get_queue().wait();
}

template <typename T>
inline void device_copy_async_dd(const T* src, T* dst, size_type count)
{
  cl::sycl::queue& q = gt::backend::sycl::get_queue();
  q.memcpy(dst, src, sizeof(T) * count);
}

} // namespace sycl

namespace copy_impl
{

template <typename InputPtr, typename OutputPtr>
inline void sycl_copy_n(InputPtr in, size_type count, OutputPtr out)
{
  cl::sycl::queue& q = gt::backend::sycl::get_queue();
  q.memcpy(gt::raw_pointer_cast(out), gt::raw_pointer_cast(in),
           sizeof(typename gt::pointer_traits<InputPtr>::element_type) * count);
  q.wait();
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
  if (element_type(value) == element_type()) {
    q.memset(gt::raw_pointer_cast(first), 0,
             (last - first) * sizeof(element_type));
  } else {
    assert(sizeof(element_type) == 1);
    q.memset(gt::raw_pointer_cast(first), value,
             (last - first) * sizeof(element_type));
  }
}
} // namespace fill_impl

} // namespace backend
} // namespace gt

#endif // GTENSOR_BACKEND_SYCL_H
