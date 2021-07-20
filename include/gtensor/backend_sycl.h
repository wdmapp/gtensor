
#ifndef GTENSOR_BACKEND_SYCL_H
#define GTENSOR_BACKEND_SYCL_H

#include <cstdlib>
#include <exception>
#include <iostream>
#include <unordered_map>

#include <CL/sycl.hpp>

#include "pointer_traits.h"

// ======================================================================
// gt::backend::sycl

namespace gt
{
namespace backend
{

namespace sycl
{

namespace detail
{

inline auto get_exception_handler()
{
  static auto exception_handler = [](cl::sycl::exception_list exceptions) {
    for (std::exception_ptr const& e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (cl::sycl::exception const& e) {
        std::cerr << "Caught asynchronous SYCL exception:" << std::endl
                  << e.what() << std::endl;
        abort();
      }
    }
  };
  return exception_handler;
}

class SyclQueues
{
public:
  SyclQueues() : current_device_id_(0)
  {
    // Get devices from default platform, which for Intel implementation
    // can be controlled with SYCL_DEVICE_FILTER env variable. This
    // allows flexible selection at runtime.
    cl::sycl::platform p{cl::sycl::default_selector()};
    // std::cout << p.get_info<cl::sycl::info::platform::name>()
    //          << " {" << p.get_info<cl::sycl::info::platform::vendor>() << "}"
    //          << std::endl;
    devices_ = p.get_devices();
  }

  cl::sycl::queue& get_queue_by_id(int device_id)
  {
    if (device_id >= devices_.size()) {
      throw std::runtime_error("No such device");
    }
    if (queue_map_.count(device_id) == 0) {
      queue_map_[device_id] =
        cl::sycl::queue{devices_[device_id], get_exception_handler()};
    }
    return queue_map_[device_id];
  }

  int get_device_count() { return devices_.size(); }

  void set_device_id(int device_id) { current_device_id_ = device_id; }

  uint32_t get_device_vendor_id(int device_id)
  {
    if (device_id >= devices_.size()) {
      throw std::runtime_error("No such device");
    }

    // TODO: this will be unique, but is not useful for it's intended
    // purpose of varifying the MPI -> GPU mapping, since it would work
    // even if the runtime returned the same device multiple times.
    return devices_[device_id].get_info<cl::sycl::info::device::vendor_id>() +
           device_id;
  }

  int get_device_id() { return current_device_id_; }

  cl::sycl::queue& get_queue() { return get_queue_by_id(current_device_id_); }

private:
  std::vector<cl::sycl::device> devices_;
  std::unordered_map<int, cl::sycl::queue> queue_map_;
  int current_device_id_;
};

inline SyclQueues& get_sycl_queues_instance()
{
  static SyclQueues queues;
  return queues;
}

} // namespace detail

/*! Get the global singleton queue object used for all device operations.  */
inline cl::sycl::queue& get_queue()
{
  return detail::get_sycl_queues_instance().get_queue();
}

inline void device_synchronize()
{
  get_queue().wait();
}

inline int device_get_count()
{
  return detail::get_sycl_queues_instance().get_device_count();
}

inline void device_set(int device_id)
{
  detail::get_sycl_queues_instance().set_device_id(device_id);
}

inline int device_get()
{
  return detail::get_sycl_queues_instance().get_device_id();
}

inline uint32_t device_get_vendor_id(int device_id)
{
  return detail::get_sycl_queues_instance().get_device_vendor_id(device_id);
}

template <typename T>
inline void device_copy_async_dd(const T* src, T* dst, size_type count)
{
  cl::sycl::queue& q = get_queue();
  q.memcpy(dst, src, sizeof(T) * count);
}

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
