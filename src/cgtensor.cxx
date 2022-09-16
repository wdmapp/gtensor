#include <stdint.h>

#include <gtensor/capi.h>
#include <gtensor/gtensor.h>

void gt_synchronize()
{
  gt::synchronize();
}

int gt_backend_device_get_count()
{
  return gt::backend::clib::device_get_count();
}

void gt_backend_device_set(int device_id)
{
  return gt::backend::clib::device_set(device_id);
}

int gt_backend_device_get()
{
  return gt::backend::clib::device_get();
}

uint32_t gt_backend_device_get_vendor_id(int device_id)
{
  return gt::backend::clib::device_get_vendor_id(device_id);
}

void* gt_backend_host_allocate(size_t nbytes)
{
  return (void*)
    gt::backend::gallocator<gt::space::clib_host>::allocate<uint8_t>(nbytes);
}

void* gt_backend_device_allocate(size_t nbytes)
{
  return (void*)
    gt::backend::gallocator<gt::space::clib_device>::allocate<uint8_t>(nbytes);
}

void* gt_backend_managed_allocate(size_t nbytes)
{
  return (void*)
    gt::backend::gallocator<gt::space::clib_managed>::allocate<uint8_t>(nbytes);
}

void gt_backend_host_deallocate(void* p)
{
  gt::backend::gallocator<gt::space::clib_host>::deallocate((uint8_t*)p);
}

void gt_backend_device_deallocate(void* p)
{
  gt::backend::gallocator<gt::space::clib_device>::deallocate((uint8_t*)p);
}

void gt_backend_managed_deallocate(void* p)
{
  gt::backend::gallocator<gt::space::clib_managed>::deallocate((uint8_t*)p);
}

#ifdef GTENSOR_HAVE_DEVICE

void gt_backend_memcpy_hh(void* dst, const void* src, size_t nbytes)
{
  gt::copy_n((const uint8_t*)src, nbytes, (uint8_t*)dst);
}

void gt_backend_memcpy_dd(void* dst, const void* src, size_t nbytes)
{
  gt::copy_n(gt::device_pointer_cast((const uint8_t*)src), nbytes,
             gt::device_pointer_cast((uint8_t*)dst));
}

void gt_backend_memcpy_async_dd(void* dst, const void* src, size_t nbytes)
{
  gt::backend::clib::copy_async_dd<uint8_t>((uint8_t*)src, (uint8_t*)dst,
                                            nbytes);
}

void gt_backend_memcpy_dh(void* dst, const void* src, size_t nbytes)
{
  gt::copy_n(gt::device_pointer_cast((const uint8_t*)src), nbytes,
             (uint8_t*)dst);
}

void gt_backend_memcpy_hd(void* dst, const void* src, size_t nbytes)
{
  gt::copy_n((const uint8_t*)src, nbytes,
             gt::device_pointer_cast((uint8_t*)dst));
}

void gt_backend_memset(void* dst, int value, size_t nbytes)
{
  gt::fill(gt::device_pointer_cast((uint8_t*)dst),
           gt::device_pointer_cast((uint8_t*)dst) + nbytes, value);
}

#endif

bool gt_backend_is_device_address(void* p)
{
  return gt::backend::clib::is_device_address(p);
}

void gt_backend_prefetch_device(void* p, size_t nbytes)
{
  gt::backend::clib::prefetch_device<uint8_t>(static_cast<uint8_t*>(p), nbytes);
}

void gt_backend_prefetch_host(void* p, size_t nbytes)
{
  gt::backend::clib::prefetch_device<uint8_t>(static_cast<uint8_t*>(p), nbytes);
}
