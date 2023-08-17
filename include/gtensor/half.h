// half_wrapper.hxx
#ifndef HALF_WRAPPER
#define HALF_WRAPPER

#include <iostream>
#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
#define TARGET_ARCH __host__ __device__
#else
#define TARGET_ARCH
#endif

class HalfWrapper
{
public:
    TARGET_ARCH HalfWrapper(float x) : x(x) {}; 
    TARGET_ARCH HalfWrapper(half x) : x(x) {}; 
    TARGET_ARCH const HalfWrapper& operator=(const float f) { x = f; return *this; }
    TARGET_ARCH const half& Get() const { return x; }
private:
    half x;
};

TARGET_ARCH const HalfWrapper operator+(const HalfWrapper& lhs, const HalfWrapper& rhs)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
return HalfWrapper( lhs.Get() + rhs.Get() );
#else
return HalfWrapper( float(lhs.Get()) + float(rhs.Get()) );
#endif
}

TARGET_ARCH const HalfWrapper operator*(const HalfWrapper& lhs, const HalfWrapper& rhs)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
return HalfWrapper( lhs.Get() * rhs.Get() );
#else
return HalfWrapper( float(lhs.Get()) * float(rhs.Get()) );
#endif
}

std::ostream& operator<<(std::ostream& s, const HalfWrapper& h)
{ s << (float) h.Get(); return s; }

#endif
