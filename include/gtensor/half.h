#ifndef GTENSOR_HALF_H
#define GTENSOR_HALF_H

#include <iostream>
#include <cuda_fp16.h>
#include <gtensor/macros.h>

namespace gt
{

// ======================================================================
// half

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
using compute_type = __half;
#else
using compute_type = float;
#endif

class half
{
public:
    half() = default;
    GT_INLINE half(float x) : x(x) {};
    GT_INLINE half(__half x) : x(x) {};

    GT_INLINE const half& operator=(const float f) { x = f; return *this; }
    GT_INLINE compute_type Get() const { return static_cast<compute_type>(x); }
private:
    __half x;
};

#define PROVIDE_HALF_BINARY_ARITHMETIC_OPERATOR(op) \
    GT_INLINE half operator op(const half& lhs, const half& rhs) \
    { return half( lhs.Get() op rhs.Get() ); }

#define PROVIDE_MIXED_HALF_BINARY_ARITHMETIC_OPERATOR(op, fp_type) \
    \
    GT_INLINE fp_type operator op(const half& lhs, const fp_type& rhs) \
    { return static_cast<fp_type>(lhs.Get()) op rhs; } \
    \
    GT_INLINE fp_type operator op(const fp_type& lhs, const half& rhs) \
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
    GT_INLINE bool operator op(const half& lhs, const half& rhs) \
    { return lhs.Get() op rhs.Get(); }

#define PROVIDE_MIXED_HALF_COMPARISON_OPERATOR(op, fp_type) \
    \
    GT_INLINE bool operator op(const half& lhs, const fp_type& rhs) \
    { return static_cast<fp_type>(lhs.Get()) op rhs; } \
    \
    GT_INLINE bool operator op(const fp_type& lhs, const half& rhs) \
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

#undef PROVIDE_HALF_BINARY_ARITHMETIC_OPERATOR
#undef PROVIDE_MIXED_HALF_BINARY_ARITHMETIC_OPERATOR
#undef PROVIDE_HALF_COMPARISON_OPERATOR
#undef PROVIDE_MIXED_HALF_COMPARISON_OPERATOR

} // namespace gt

#endif
