#ifndef GTENSOR_HALF_H
#define GTENSOR_HALF_H

#include <iostream>
#include <cuda_fp16.h>

namespace gt
{

// ======================================================================
// half
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
#define TARGET_ARCH __host__ __device__
#else
#define TARGET_ARCH
#endif

class half
{
public:
    TARGET_ARCH half(float x) : x(x) {};
    TARGET_ARCH half(__half x) : x(x) {};
    TARGET_ARCH const half& operator=(const float f) { x = f; return *this; }
    TARGET_ARCH const __half& Get() const { return x; }
private:
    __half x;
};

TARGET_ARCH const half operator+(const half& lhs, const half& rhs)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return half( lhs.Get() + rhs.Get() );
#else
    return half( float(lhs.Get()) + float(rhs.Get()) );
#endif
}

TARGET_ARCH const half operator*(const half& lhs, const half& rhs)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return half( lhs.Get() * rhs.Get() );
#else
    return half( float(lhs.Get()) * float(rhs.Get()) );
#endif
}

std::ostream& operator<<(std::ostream& s, const half& h)
{ s << (float) h.Get(); return s; }

} // namespace gt

#endif
