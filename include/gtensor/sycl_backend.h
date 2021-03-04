#ifndef GTENSOR_SYCL_BACKEND_H
#define GTENSOR_SYCL_BACKEND_H

#include <cstdlib>
#include <iostream>

#include <CL/sycl.hpp>

namespace gt
{
namespace backend
{
namespace sycl
{

inline cl::sycl::device& get_device()
{
#ifdef GTENSOR_DEVICE_SYCL_SELECTOR_GPU
  static cl::sycl::device d{cl::sycl::gpu_selector()};
#elif defined(GTENSOR_DEVICE_SYCL_SELECTOR_CPU)
  static cl::sycl::device d{cl::sycl::cpu_selector()};
#elif defined(GTENSOR_DEVICE_SYCL_SELECTOR_HOST)
  static cl::sycl::device d{cl::sycl::host_selector()};
#else
  static cl::sycl::device d{cl::sycl::default_selector()};
#endif
  return d;
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

  static cl::sycl::queue q{get_device(), exception_handler};
  return q;
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
