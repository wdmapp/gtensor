#ifndef GTENSOR_SYCL_BACKEND_H
#define GTENSOR_SYCL_BACKEND_H

#define GTENSOR_SYCL_BACKEND_MAX_DEVICES 10

#include <cstdlib>
#include <exception>
#include <iostream>
#include <unordered_map>

#include <CL/sycl.hpp>

namespace gt
{
namespace backend
{
namespace sycl
{

inline auto get_exception_handler_()
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

class SyclQueues_
{
public:
  SyclQueues_() : current_device_id_(0)
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
        cl::sycl::queue{devices_[device_id], get_exception_handler_()};
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

inline SyclQueues_& get_sycl_queues_instance_()
{
  static SyclQueues_ queues;
  return queues;
}

/*! Get the global singleton queue object used for all device operations.
 *
 * TODO: allow lib clients to customize this, via a cuda-like API
 * e.g. sylcSetDevice(int), syclGetDevice(int), for multi-gpu systems.
 * The idea is to support one device per MPI process use cases, and
 * not worry about more complex cases.
 */
inline cl::sycl::queue& get_queue()
{
  return get_sycl_queues_instance_().get_queue();
}

inline void set_device(int device_id)
{
  get_sycl_queues_instance_().set_device_id(device_id);
}

inline int get_device_count()
{
  return get_sycl_queues_instance_().get_device_count();
}

inline int get_device()
{
  return get_sycl_queues_instance_().get_device_id();
}

inline int get_device_vendor_id(int device_id)
{
  return get_sycl_queues_instance_().get_device_vendor_id(device_id);
}

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
} // namespace backend
} // namespace gt

#endif // GENSOR_SYCL_BACKEND_H
