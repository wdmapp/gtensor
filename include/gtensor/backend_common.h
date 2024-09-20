#ifndef GTENSOR_BACKEND_COMMON_H
#define GTENSOR_BACKEND_COMMON_H

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <type_traits>

#ifdef GTENSOR_HAVE_DEVICE

#ifdef GTENSOR_HAVE_THRUST
#include "thrust_ext.h"
#include <thrust/device_vector.h>
#endif

#endif // GTENSOR_HAVE_DEVICE

#include "defs.h"
#include "macros.h"
#include "pointer_traits.h"
#include "space_forward.h"

#ifdef GTENSOR_USE_MEMORY_POOL
#include "memory_pool.h"
#endif

namespace gt
{

namespace backend
{

enum class memory_type
{
  host,
  device,
  managed,
  unregistered,
};

// ======================================================================
// library wide configuration

#ifndef GTENSOR_MANAGED_MEMORY_TYPE_DEFAULT
#define GTENSOR_MANAGED_MEMORY_TYPE_DEFAULT managed
#endif

#ifdef GTENSOR_DEVICE_HIP
#if HIP_VERSION_MAJOR >= 5
enum class managed_memory_type
{
  managed,
  device,
  managed_fine,
  managed_coarse
};
#else
enum class managed_memory_type
{
  managed,
  device,
  managed_fine
};
#endif // HIP_VERSION_MAJOR

#if HIP_VERSION_MAJOR >= 6 || (HIP_VERSION_MAJOR == 5 && HIP_VERSION_MINOR >= 5)
#define GTENSOR_HIP_NEW_INCLUDES
#endif
#else
enum class managed_memory_type
{
  managed,
  device
};
#endif

namespace config
{

#define QUALIFY_MMTYPE(x) gt::backend::managed_memory_type::x

struct gtensor_config
{
  gt::backend::managed_memory_type managed_memory_type =
    QUALIFY_MMTYPE(GTENSOR_MANAGED_MEMORY_TYPE_DEFAULT);
};

#undef QUALIFY_MMTYPE

inline gtensor_config& get_instance()
{
  static gtensor_config config;
  return config;
}

} // namespace config

inline void set_managed_memory_type(managed_memory_type mtype)
{
  config::get_instance().managed_memory_type = mtype;
}

inline auto get_managed_memory_type()
{
  return config::get_instance().managed_memory_type;
}

// ======================================================================
// stream interface

namespace stream_interface
{

template <typename Stream>
Stream create();

template <typename Stream>
void destroy(Stream s);

/**
 * CRTP static interface for backend specific stream wrapper, non owning view.
 *
 * View type:
 * - reference semantics
 * - two constructors - default creates a view to default stream, alternate
 *   form takes a backend stream and creates a view from it.
 */
template <typename Stream>
class stream_view_base
{
public:
  using stream_t = Stream;

  stream_view_base(stream_t s) : stream_(s) {}

  stream_t& get_backend_stream() { return stream_; }

protected:
  Stream stream_;
};

/**
 * CRTP static interface for backend specific stream wrappers. This must be
 * used to define an owning type with RIAA.
 *
 * Owning type:
 * - copying not allowed
 * - one default constructor, creates a new stream
 *
 * Backend implementations should inherit constructors, e.g.
 *
 *   using base_type = stream_base<...>;
 *   using base_type::base_type;
 */
template <typename Stream, typename View = stream_view_base<Stream>>
class stream_base
{
public:
  using stream_t = Stream;
  using view_t = View;

  stream_base() : view_(gt::backend::stream_interface::create<stream_t>()) {}
  ~stream_base() { sync_and_destroy(view_.get_backend_stream()); }

  // copy not allowed
  stream_base(stream_base& other) = delete;
  stream_base& operator=(const stream_base& other) = delete;

  stream_base(stream_base&& other) : view_(other.view_)
  {
    other.moved_from_ = true;
  }

  stream_base& operator=(stream_base&& other)
  {
    stream_t old_stream = view_.get_backend_stream();
    view_ = other.view_;
    other.moved_from_ = true;
    sync_and_destroy(old_stream);
  }

  view_t& get_view() { return view_; }

  stream_t get_backend_stream() { return view_.get_backend_stream(); }

  bool is_default() { return view_.is_default(); }

  void synchronize() { view_.synchronize(); }

protected:
  bool moved_from_;
  view_t view_;

  void sync_and_destroy(stream_t s)
  {
    if (!moved_from_) {
      gt::backend::stream_interface::destroy<stream_t>(s);
    }
  }
};

} // namespace stream_interface

// ======================================================================
// space_pointer_impl::selector

namespace space_pointer_impl
{
template <typename S>
struct selector
{
  template <typename T>
  using pointer = gt::backend::device_ptr<T>;
};

template <>
struct selector<gt::space::host>
{
  template <typename T>
  using pointer = T*;
};

#ifdef GTENSOR_HAVE_THRUST
template <>
struct selector<gt::space::thrust>
{
  template <typename T>
  using pointer = ::thrust::device_ptr<T>;
};
#endif

} // namespace space_pointer_impl

} // namespace backend

template <typename T, typename S>
using space_pointer =
  typename gt::backend::space_pointer_impl::selector<S>::template pointer<T>;

template <typename T, typename S = gt::space::device>
using device_ptr = space_pointer<T, S>;

template <typename T, typename S = gt::space::host>
using host_ptr = space_pointer<T, S>;

template <typename P>
GT_INLINE auto raw_pointer_cast(P p)
{
  return gt::pointer_traits<P>::get(p);
}

template <typename T>
GT_INLINE auto device_pointer_cast(T* p)
{
  using pointer = typename gt::device_ptr<T, gt::space::device>;
  return pointer(p);
}

template <typename S = gt::space::device, typename T>
GT_INLINE auto space_pointer_cast(T* p)
{
  using pointer = typename gt::device_ptr<T, S>;
  return pointer(p);
}

namespace backend
{
namespace allocator_impl
{

template <typename T, typename A, typename S>
struct wrap_allocator
{
  using value_type = T;
  using pointer = gt::space_pointer<T, S>;
  using size_type = gt::size_type;

  pointer allocate(size_type n) { return pointer(A::template allocate<T>(n)); }
  void deallocate(pointer p, size_type n)
  {
    A::deallocate(gt::pointer_traits<pointer>::get(p));
  }
};

template <typename S>
struct gallocator;

template <typename T, typename S>
struct selector
{
  using type = wrap_allocator<T, gallocator<S>, S>;
};

#ifdef GTENSOR_USE_MEMORY_POOL

template <typename Space, gt::memory_pool::memory_type MemType>
struct pool_gallocator
{
  using space_type = Space;

  template <typename T>
  static T* allocate(size_type n)
  {
    auto nbytes = sizeof(T) * n;
    return static_cast<T*>(
      gt::memory_pool::get_instance().allocate<MemType>(nbytes));
  }

  template <typename T>
  static void deallocate(T* p)
  {
    gt::memory_pool::get_instance().deallocate<MemType>(p);
  }
};

template <typename Space>
struct pool_gallocator<Space, gt::memory_pool::memory_type::managed>
{
  using space_type = Space;

  template <typename T>
  static T* allocate(size_type n)
  {
    auto nbytes = sizeof(T) * n;
    auto mtype = gt::backend::get_managed_memory_type();
    if (mtype == gt::backend::managed_memory_type::device) {
      return static_cast<T*>(
        gt::memory_pool::get_instance()
          .allocate<gt::memory_pool::memory_type::device>(nbytes));
    } else {
      // Note: HIP specialization calls this for managed_* types
      return static_cast<T*>(
        gt::memory_pool::get_instance()
          .allocate<gt::memory_pool::memory_type::managed>(nbytes));
    }
  }

  template <typename T>
  static void deallocate(T* p)
  {
    auto mtype = gt::backend::get_managed_memory_type();
    if (mtype == gt::backend::managed_memory_type::managed) {
      gt::memory_pool::get_instance()
        .deallocate<gt::memory_pool::memory_type::managed>(p);
    } else if (mtype == gt::backend::managed_memory_type::device) {
      gt::memory_pool::get_instance()
        .deallocate<gt::memory_pool::memory_type::device>(p);
    } else {
      throw std::runtime_error("unsupported managed memory type for backend");
    }
  }
};

#endif // GTENSOR_USE_MEMORY_POOL

} // namespace allocator_impl

template <typename S>
class backend_ops;

} // namespace backend

} // namespace gt

#endif // GTENSOR_BACKEND_COMMON_H
