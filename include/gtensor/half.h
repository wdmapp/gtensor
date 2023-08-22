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
using compute_type = __half;
#else
#define TARGET_ARCH
using compute_type = float;
#endif

class half
{
public:
    half() = default;
    TARGET_ARCH half(float x) : x(x) {};
    TARGET_ARCH half(__half x) : x(x) {};

    TARGET_ARCH const half& operator=(const float f) { x = f; return *this; }
    TARGET_ARCH compute_type Get() const { return static_cast<compute_type>(x); }
private:
    __half x;
};

#define PROVIDE_HALF_BINARY_ARITHMETIC_OPERATOR(op) \
    TARGET_ARCH const half operator op(const half& lhs, const half& rhs) \
    { return half( lhs.Get() op rhs.Get() ); }

PROVIDE_HALF_BINARY_ARITHMETIC_OPERATOR(+);
PROVIDE_HALF_BINARY_ARITHMETIC_OPERATOR(-);
PROVIDE_HALF_BINARY_ARITHMETIC_OPERATOR(*);
PROVIDE_HALF_BINARY_ARITHMETIC_OPERATOR(/);

TARGET_ARCH bool operator==(const half& lhs, const half& rhs)
{
    return lhs.Get() == rhs.Get();
}

TARGET_ARCH bool operator!=(const half& lhs, const half& rhs)
{
    return !(lhs == rhs);
}

TARGET_ARCH bool operator<(const half& lhs, const half& rhs)
{
    return lhs.Get() < rhs.Get();
}

TARGET_ARCH bool operator<=(const half& lhs, const half& rhs)
{
    return lhs.Get() <= rhs.Get();
}

TARGET_ARCH bool operator>(const half& lhs, const half& rhs)
{
    return lhs.Get() > rhs.Get();
}

TARGET_ARCH bool operator>=(const half& lhs, const half& rhs)
{
    return lhs.Get() >= rhs.Get();
}

std::ostream& operator<<(std::ostream& s, const half& h)
{ s << (float) h.Get(); return s; }

} // namespace gt

#endif
