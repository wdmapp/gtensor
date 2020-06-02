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

} // namespace sycl
} // namespace backend
} // namespace gt

#endif // GENSOR_SYCL_BACKEND_H
