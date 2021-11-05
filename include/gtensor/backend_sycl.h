
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

// fallback if none of the backend specific methods succeed
inline uint32_t get_unique_device_id_sycl(int device_index,
                                          const cl::sycl::device& d)
{
  // TODO: this will be unique, but is not useful for it's intended
  // purpose of varifying the MPI -> GPU mapping, since it would work
  // even if the runtime returned the same device multiple times.
  return d.get_info<cl::sycl::info::device::vendor_id>() + device_index;
}

template <cl::sycl::backend Backend>
uint32_t get_unique_device_id(int device_index, const cl::sycl::device& d);

#ifdef GTENSOR_DEVICE_SYCL_OPENCL
typedef struct _cl_device_pci_bus_info_khr
{
  cl_uint pci_domain;
  cl_uint pci_bus;
  cl_uint pci_device;
  cl_uint pci_function;
} cl_device_pci_bus_info_khr;
#define CL_DEVICE_PCI_BUS_INFO_KHR 0x410F

template <>
inline uint32_t get_unique_device_id<cl::sycl::backend::opencl>(
  int device_index, const cl::sycl::device& d)
{
  uint32_t unique_id = 0;
  cl_device_id cl_dev = d.get_native<cl::sycl::backend::opencl>();
  cl_device_pci_bus_info_khr pci_info;
  cl_int rval = clGetDeviceInfo(cl_dev, CL_DEVICE_PCI_BUS_INFO_KHR,
                                sizeof(pci_info), &pci_info, NULL);
  if (rval == CL_SUCCESS) {
    unique_id |= (0x000000FF & (pci_info.pci_device));
    unique_id |= (0x0000FF00 & (pci_info.pci_bus << 8));
    unique_id |= (0xFFFF0000 & (pci_info.pci_domain << 16));
    // std::cout << "opencl (pci_bus_info ext) " << unique_id << std::endl;
  }
  if (unique_id == 0) {
    unique_id = get_unique_device_id_sycl(device_index, d);
    // std::cout << "opencl (sycl fallback) " << unique_id << std::endl;
  }
  return unique_id;
}
#endif

#ifdef GTENSOR_DEVICE_SYCL_L0
template <>
inline uint32_t get_unique_device_id<cl::sycl::backend::level_zero>(
  int device_index, const cl::sycl::device& d)
{
  uint32_t unique_id = 0;

  ze_device_handle_t ze_dev = d.get_native<cl::sycl::backend::level_zero>();
  ze_device_properties_t ze_prop;
  zeDeviceGetProperties(ze_dev, &ze_prop);

  // Try to use Level Zero Sysman API to get PCI id. Requires setting
  // ZES_ENABLE_SYSMAN=1 in the environment to enable.
  zes_device_handle_t zes_dev = reinterpret_cast<zes_device_handle_t>(ze_dev);
  zes_pci_properties_t pci_props;
  if (zesDevicePciGetProperties(zes_dev, &pci_props) == ZE_RESULT_SUCCESS) {
    unique_id |= (0x000000FF & (pci_props.address.device));
    unique_id |= (0x0000FF00 & (pci_props.address.bus << 8));
    unique_id |= (0xFFFF0000 & (pci_props.address.domain << 16));
    // std::cout << "level zero (sysman): " << unique_id << std::endl;
  }

  // try device uuid first 4 bytes
  if (unique_id == 0) {
    unique_id |= (0x000000FF & ze_prop.uuid.id[3]);
    unique_id |= (0x0000FF00 & ze_prop.uuid.id[2]);
    unique_id |= (0x00FF0000 & ze_prop.uuid.id[1]);
    unique_id |= (0xFF000000 & ze_prop.uuid.id[0]);
    // std::cout << "level zero (uuid): " << unique_id << std::endl;
  }

  // if sysman and uuid fails, try the vendorId and deviceid. This is not
  // unique yet in intel implementation, so add device index.
  if (unique_id == 0) {
    unique_id = (ze_prop.vendorId << 16) + ze_prop.deviceId + device_index;
    // std::cout << "level zero (deviceId + idx): " << unique_id << std::endl;
  }

  // last resort, fallback to pure SYCL method
  if (unique_id == 0) {
    unique_id = get_unique_device_id_sycl(device_index, d);
    // std::cout << "level zero (sycl fallback): " << unique_id << std::endl;
  }

  return unique_id;
}
#endif

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

  cl::sycl::queue new_queue(int device_id)
  {
    if (device_id >= devices_.size()) {
      throw std::runtime_error("No such device");
    }
    return cl::sycl::queue{devices_[device_id], get_exception_handler()};
  }

  cl::sycl::queue new_queue() { return new_queue(current_device_id_); }

  int get_device_count() { return devices_.size(); }

  void set_device_id(int device_id) { current_device_id_ = device_id; }

  uint32_t get_device_vendor_id(int device_id)
  {
    if (device_id >= devices_.size()) {
      throw std::runtime_error("No such device");
    }

    const cl::sycl::device& sycl_dev = devices_[device_id];
    const cl::sycl::platform& p = sycl_dev.get_platform();
    std::string p_name = p.get_info<cl::sycl::info::platform::name>();
    if (false) {
#ifdef GTENSOR_DEVICE_SYCL_L0
    } else if (p_name.find("Level-Zero") != std::string::npos) {
      return get_unique_device_id<cl::sycl::backend::level_zero>(device_id,
                                                                 sycl_dev);
#endif
#ifdef GTENSOR_DEVICE_SYCL_OPENCL
    } else if (p_name.find("OpenCL") != std::string::npos) {
      return get_unique_device_id<cl::sycl::backend::opencl>(device_id,
                                                             sycl_dev);
#endif
    } else {
      return get_unique_device_id_sycl(device_id, sycl_dev);
    }
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

/*! Get a new queue different from the default, for use like alternate streams.
 */
inline cl::sycl::queue new_queue()
{
  return detail::get_sycl_queues_instance().new_queue();
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

class stream_view
{
public:
  stream_view() : stream_(gt::backend::sycl::get_queue()) {}
  stream_view(cl::sycl::queue q) : stream_(q) {}

  auto get_backend_stream() { return stream_; }

  bool is_default() { return stream_ == gt::backend::sycl::get_queue(); }

  void synchronize() { stream_.wait(); }

private:
  cl::sycl::queue stream_;
};

class stream_wrapper
{
public:
  stream_wrapper() { stream_ = gt::backend::sycl::new_queue(); }

  auto get_backend_stream() { return stream_; }

  bool is_default() { return stream_ == gt::backend::sycl::get_queue(); }

  auto get_view() { return stream_view(stream_); }

  void synchronize() { stream_.wait(); }

private:
  cl::sycl::queue stream_;
};

} // namespace gt

#endif // GTENSOR_BACKEND_SYCL_H
