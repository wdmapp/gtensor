#ifndef GTENSOR_SYCL_BACKEND_H
#define GTENSOR_SYCL_BACKEND_H

#include <CL/sycl.hpp>

namespace gt
{
namespace backend
{
namespace sycl
{

/*! Get the global singleton queue object used for all device operations.
 *
 * TODO: allow lib clients to customize this, via a cuda-like API
 * e.g. sylcSetDevice(int), syclGetDevice(int), for multi-gpu systems.
 * The idea is to support one device per MPI process use cases, and
 * not worry about more complex cases.
 */
static inline cl::sycl::queue& get_queue()
{
  static cl::sycl::queue q{};
  return q;
}

template <typename E1, typename E2>
class Assign1;
template <typename E1, typename E2>
class Assign2;
template <typename E1, typename E2>
class Assign3;
template <typename E1, typename E2>
class AssignN;

template <typename F>
class Launch1;
template <typename F>
class Launch2;
template <typename F>
class Launch3;
template <typename F>
class LaunchN;

} // namespace sycl
} // namespace backend
} // namespace gt

#endif // GENSOR_SYCL_BACKEND_H
