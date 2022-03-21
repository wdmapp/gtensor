#ifndef GTENSOR_BACKEND_SYCL_DEVICE_H
#define GTENSOR_BACKEND_SYCL_DEVICE_H

#include <cstdlib>
#include <exception>
#include <forward_list>
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

// ======================================================================
// gt::backend::sycl::device

namespace gt
{
namespace backend
{

namespace sycl
{

namespace device
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
  cl_device_id cl_dev = cl::sycl::get_native<cl::sycl::backend::opencl>(d);
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
inline uint32_t get_unique_device_id<cl::sycl::backend::ext_oneapi_level_zero>(
  int device_index, const cl::sycl::device& d)
{
  uint32_t unique_id = 0;

  ze_device_handle_t ze_dev =
    cl::sycl::get_native<cl::sycl::backend::ext_oneapi_level_zero>(d);
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
    if (ze_prop.flags & ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE) {
      unique_id += ze_prop.subdeviceId;
    }
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

inline bool device_per_tile_enabled()
{
  static bool init = false;
  static bool enabled;
  if (!init) {
    enabled =
      (std::getenv("GTENSOR_DEVICE_SYCL_DISABLE_SUB_DEVICES") == nullptr);
  }
  return enabled;
}

inline std::vector<cl::sycl::device> get_devices_with_numa_sub(
  const cl::sycl::platform& p)
{
  std::vector<cl::sycl::device> result;
  for (auto root_dev : p.get_devices()) {
    // Handle GPUs with multiple tiles, which can be partitioned based on numa
    // domain
    auto max_sub_devices =
      root_dev.get_info<cl::sycl::info::device::partition_max_sub_devices>();
    // NB: workaround bug in host backend, where max > 0 but it's not supported
    if (device_per_tile_enabled() && max_sub_devices > 0 &&
        (root_dev.is_gpu() || root_dev.is_cpu())) {
      auto sub_devices = root_dev.create_sub_devices<
        cl::sycl::info::partition_property::partition_by_affinity_domain>(
        cl::sycl::info::partition_affinity_domain::numa);
      for (auto sub_dev : sub_devices) {
        result.push_back(sub_dev);
      }
    } else {
      result.push_back(root_dev);
    }
  }
  return result;
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
    devices_ = get_devices_with_numa_sub(p);
  }

  void valid_device_id_or_throw(int device_id)
  {
    if (device_id >= devices_.size() || device_id < 0) {
      throw std::runtime_error("No such device");
    }
  }

  cl::sycl::queue& get_queue(int device_id)
  {
    valid_device_id_or_throw(device_id);
    if (default_queue_map_.count(device_id) == 0) {
      default_queue_map_[device_id] =
        cl::sycl::queue{devices_[device_id], get_exception_handler(),
                        cl::sycl::property::queue::in_order()};
    }
    return default_queue_map_[device_id];
  }

  cl::sycl::queue& new_stream_queue(int device_id)
  {
    valid_device_id_or_throw(device_id);
    stream_queue_map_[device_id].emplace_front(
      devices_[device_id], get_exception_handler(),
      cl::sycl::property::queue::in_order());
    return stream_queue_map_[device_id].front();
  }

  cl::sycl::queue& new_stream_queue()
  {
    return new_stream_queue(current_device_id_);
  }

  void delete_stream_queue(int device_id, cl::sycl::queue& q)
  {
    valid_device_id_or_throw(device_id);
    stream_queue_map_[device_id].remove(q);
  }

  void delete_stream_queue(cl::sycl::queue& q)
  {
    delete_stream_queue(current_device_id_, q);
  }

  int get_device_count() { return devices_.size(); }

  void set_device_id(int device_id)
  {
    valid_device_id_or_throw(device_id);
    current_device_id_ = device_id;
  }

  uint32_t get_device_vendor_id(int device_id)
  {
    valid_device_id_or_throw(device_id);
    const cl::sycl::device& sycl_dev = devices_[device_id];
    const cl::sycl::platform& p = sycl_dev.get_platform();
    std::string p_name = p.get_info<cl::sycl::info::platform::name>();
    if (false) {
#ifdef GTENSOR_DEVICE_SYCL_L0
    } else if (p_name.find("Level-Zero") != std::string::npos) {
      return get_unique_device_id<cl::sycl::backend::ext_oneapi_level_zero>(
        device_id, sycl_dev);
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

  bool has_open_stream_queues(int device_id)
  {
    return !stream_queue_map_[device_id].empty();
  }

  bool has_open_stream_queues()
  {
    return has_open_stream_queues(current_device_id_);
  }

  cl::sycl::queue& get_queue() { return get_queue(current_device_id_); }

private:
  std::vector<cl::sycl::device> devices_;
  std::unordered_map<int, cl::sycl::queue> default_queue_map_;
  std::unordered_map<int, std::forward_list<cl::sycl::queue>> stream_queue_map_;
  int current_device_id_;
};

inline SyclQueues& get_sycl_queues_instance()
{
  static SyclQueues queues;
  return queues;
}

} // namespace device

/*! Get the global singleton queue object used for all device operations.  */
inline cl::sycl::queue& get_queue()
{
  return device::get_sycl_queues_instance().get_queue();
}

inline cl::sycl::queue& get_queue(int device_id)
{
  return device::get_sycl_queues_instance().get_queue(device_id);
}

/*! Get a new queue different from the default, for use like alternate streams.
 */
inline cl::sycl::queue& new_stream_queue()
{
  return device::get_sycl_queues_instance().new_stream_queue();
}

inline cl::sycl::queue& new_stream_queue(int device_id)
{
  return device::get_sycl_queues_instance().new_stream_queue(device_id);
}

/*! Get a new queue different from the default, for use like alternate streams.
 */
inline void delete_stream_queue(cl::sycl::queue& q)
{
  device::get_sycl_queues_instance().delete_stream_queue(q);
}

inline void delete_stream_queue(int device_id, cl::sycl::queue& q)
{
  device::get_sycl_queues_instance().delete_stream_queue(device_id, q);
}

/*! Get a new queue different from the default, for use like alternate streams.
 */
inline bool has_open_stream_queues(int device_id)
{
  return device::get_sycl_queues_instance().has_open_stream_queues(device_id);
}

inline bool has_open_stream_queues()
{
  return device::get_sycl_queues_instance().has_open_stream_queues();
}

} // namespace sycl

} // namespace backend

} // namespace gt

#endif // GTENSOR_BACKEND_SYCL_DEVICE_H
