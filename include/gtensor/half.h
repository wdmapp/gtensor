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
    TARGET_ARCH half operator op(const half& lhs, const half& rhs) \
    { return half( lhs.Get() op rhs.Get() ); }

#define PROVIDE_MIXED_HALF_BINARY_ARITHMETIC_OPERATOR(op, fp_type) \
    \
    TARGET_ARCH fp_type operator op(const half& lhs, const fp_type& rhs) \
    { return static_cast<fp_type>(lhs.Get()) op rhs; } \
    \
    TARGET_ARCH fp_type operator op(const fp_type& lhs, const half& rhs) \
    { return lhs op static_cast<fp_type>(rhs.Get()); }

PROVIDE_HALF_BINARY_ARITHMETIC_OPERATOR(+);
PROVIDE_HALF_BINARY_ARITHMETIC_OPERATOR(-);
PROVIDE_HALF_BINARY_ARITHMETIC_OPERATOR(*);
PROVIDE_HALF_BINARY_ARITHMETIC_OPERATOR(/);

PROVIDE_MIXED_HALF_BINARY_ARITHMETIC_OPERATOR(+, float);
PROVIDE_MIXED_HALF_BINARY_ARITHMETIC_OPERATOR(-, float);
PROVIDE_MIXED_HALF_BINARY_ARITHMETIC_OPERATOR(*, float);
PROVIDE_MIXED_HALF_BINARY_ARITHMETIC_OPERATOR(/, float);

PROVIDE_MIXED_HALF_BINARY_ARITHMETIC_OPERATOR(+, double);
PROVIDE_MIXED_HALF_BINARY_ARITHMETIC_OPERATOR(-, double);
PROVIDE_MIXED_HALF_BINARY_ARITHMETIC_OPERATOR(*, double);
PROVIDE_MIXED_HALF_BINARY_ARITHMETIC_OPERATOR(/, double);

#define PROVIDE_HALF_COMPARISON_OPERATOR(op) \
    TARGET_ARCH bool operator op(const half& lhs, const half& rhs) \
    { return lhs.Get() op rhs.Get(); }

#define PROVIDE_MIXED_HALF_COMPARISON_OPERATOR(op, fp_type) \
    \
    TARGET_ARCH bool operator op(const half& lhs, const fp_type& rhs) \
    { return static_cast<fp_type>(lhs.Get()) op rhs; } \
    \
    TARGET_ARCH bool operator op(const fp_type& lhs, const half& rhs) \
    { return lhs op static_cast<fp_type>(rhs.Get()); }

PROVIDE_HALF_COMPARISON_OPERATOR(==);
PROVIDE_HALF_COMPARISON_OPERATOR(!=);
PROVIDE_HALF_COMPARISON_OPERATOR(<);
PROVIDE_HALF_COMPARISON_OPERATOR(<=);
PROVIDE_HALF_COMPARISON_OPERATOR(>);
PROVIDE_HALF_COMPARISON_OPERATOR(>=);

PROVIDE_MIXED_HALF_COMPARISON_OPERATOR(==, float);
PROVIDE_MIXED_HALF_COMPARISON_OPERATOR(!=, float);
PROVIDE_MIXED_HALF_COMPARISON_OPERATOR(<, float);
PROVIDE_MIXED_HALF_COMPARISON_OPERATOR(<=, float);
PROVIDE_MIXED_HALF_COMPARISON_OPERATOR(>, float);
PROVIDE_MIXED_HALF_COMPARISON_OPERATOR(>=, float);

PROVIDE_MIXED_HALF_COMPARISON_OPERATOR(==, double);
PROVIDE_MIXED_HALF_COMPARISON_OPERATOR(!=, double);
PROVIDE_MIXED_HALF_COMPARISON_OPERATOR(<, double);
PROVIDE_MIXED_HALF_COMPARISON_OPERATOR(<=, double);
PROVIDE_MIXED_HALF_COMPARISON_OPERATOR(>, double);
PROVIDE_MIXED_HALF_COMPARISON_OPERATOR(>=, double);

std::ostream& operator<<(std::ostream& s, const half& h)
{ s << static_cast<float>(h.Get()); return s; }

#undef TARGET_ARCH
#undef PROVIDE_HALF_BINARY_ARITHMETIC_OPERATOR
#undef PROVIDE_MIXED_HALF_BINARY_ARITHMETIC_OPERATOR
#undef PROVIDE_HALF_COMPARISON_OPERATOR
#undef PROVIDE_MIXED_HALF_COMPARISON_OPERATOR

} // namespace gt

#endif
