#ifndef GTENSOR_BFLOAT16T_H
#define GTENSOR_BFLOAT16T_H

#include <cmath>
#include <iostream>

#if __has_include(<cuda_bf16.h>)
#include <cuda_bf16.h>
#define GTENSOR_BF16_CUDA_HEADER
#elif 0 // TODO check if other bf16 type available
#else
#error "GTENSOR_ENABLE_BF16=ON, but no bfloat16 type available!"
#endif

#include "macros.h"

namespace gt
{

// ======================================================================
// bfloat16_t

class bfloat16_t
{

#if defined(GTENSOR_BF16_CUDA_HEADER)
using storage_type = __nv_bfloat16;
#else
#error "GTENSOR_ENABLE_BF16=ON, but no bfloat16 type available!"
#endif

#if defined(GTENSOR_BF16_CUDA_HEADER) && defined(__CUDA_ARCH__) &&             \
  (__CUDA_ARCH__ >= 800)
using compute_type = __nv_bfloat16;
#define BFLOAT16T_ON_CUDA_DEVICE
#else
using compute_type = float;
#endif

public:
  bfloat16_t() = default;
  GT_INLINE bfloat16_t(float x) : x(x){};
  GT_INLINE bfloat16_t(storage_type x) : x(x){};

  GT_INLINE const bfloat16_t& operator=(const float f)
  {
    x = f;
    return *this;
  }
  GT_INLINE compute_type Get() const { return static_cast<compute_type>(x); }

  // update operators [+=, -=, *=, /=]
  GT_INLINE bfloat16_t operator+=(const bfloat16_t& y)
  {
#if defined(BFLOAT16T_ON_CUDA_DEVICE)
    x += y.Get();
#else
    x = this->Get() + y.Get();
#endif
    return *this;
  }
  GT_INLINE bfloat16_t operator-=(const bfloat16_t& y)
  {
#if defined(BFLOAT16T_ON_CUDA_DEVICE)
    x -= y.Get();
#else
    x = this->Get() - y.Get();
#endif
    return *this;
  }
  GT_INLINE bfloat16_t operator*=(const bfloat16_t& y)
  {
#if defined(BFLOAT16T_ON_CUDA_DEVICE)
    x *= y.Get();
#else
    x = this->Get() * y.Get();
#endif
    return *this;
  }
  GT_INLINE bfloat16_t operator/=(const bfloat16_t& y)
  {
#if defined(BFLOAT16T_ON_CUDA_DEVICE)
    x /= y.Get();
#else
    x = this->Get() / y.Get();
#endif
    return *this;
  }

private:
  storage_type x;
};

// op is unary [+, -]
#define PROVIDE_BFLOAT16T_UNARY_ARITHMETIC_OPERATOR(op)                        \
  GT_INLINE bfloat16_t operator op(const bfloat16_t& rhs)                      \
  {                                                                            \
    return bfloat16_t(op rhs.Get());                                           \
  }

PROVIDE_BFLOAT16T_UNARY_ARITHMETIC_OPERATOR(+);
PROVIDE_BFLOAT16T_UNARY_ARITHMETIC_OPERATOR(-);

// op is binary [+, -, *, /]
#define PROVIDE_BFLOAT16T_BINARY_ARITHMETIC_OPERATOR(op)                       \
  GT_INLINE bfloat16_t operator op(const bfloat16_t& lhs,                      \
                                   const bfloat16_t& rhs)                      \
  {                                                                            \
    return bfloat16_t(lhs.Get() op rhs.Get());                                 \
  }

PROVIDE_BFLOAT16T_BINARY_ARITHMETIC_OPERATOR(+);
PROVIDE_BFLOAT16T_BINARY_ARITHMETIC_OPERATOR(-);
PROVIDE_BFLOAT16T_BINARY_ARITHMETIC_OPERATOR(*);
PROVIDE_BFLOAT16T_BINARY_ARITHMETIC_OPERATOR(/);

// op is binary [+, -, *, /]
// fp_type is [float, double]
#define PROVIDE_MIXED_BFLOAT16T_BINARY_ARITHMETIC_OPERATOR(op, fp_type)        \
                                                                               \
  GT_INLINE fp_type operator op(const bfloat16_t& lhs, const fp_type& rhs)     \
  {                                                                            \
    return static_cast<fp_type>(lhs.Get()) op rhs;                             \
  }                                                                            \
                                                                               \
  GT_INLINE fp_type operator op(const fp_type& lhs, const bfloat16_t& rhs)     \
  {                                                                            \
    return lhs op static_cast<fp_type>(rhs.Get());                             \
  }

PROVIDE_MIXED_BFLOAT16T_BINARY_ARITHMETIC_OPERATOR(+, float);
PROVIDE_MIXED_BFLOAT16T_BINARY_ARITHMETIC_OPERATOR(-, float);
PROVIDE_MIXED_BFLOAT16T_BINARY_ARITHMETIC_OPERATOR(*, float);
PROVIDE_MIXED_BFLOAT16T_BINARY_ARITHMETIC_OPERATOR(/, float);

PROVIDE_MIXED_BFLOAT16T_BINARY_ARITHMETIC_OPERATOR(+, double);
PROVIDE_MIXED_BFLOAT16T_BINARY_ARITHMETIC_OPERATOR(-, double);
PROVIDE_MIXED_BFLOAT16T_BINARY_ARITHMETIC_OPERATOR(*, double);
PROVIDE_MIXED_BFLOAT16T_BINARY_ARITHMETIC_OPERATOR(/, double);

// op is binary [==, !=, <, <=, >, >=]
#define PROVIDE_BFLOAT16T_COMPARISON_OPERATOR(op)                              \
  GT_INLINE bool operator op(const bfloat16_t& lhs, const bfloat16_t& rhs)     \
  {                                                                            \
    return lhs.Get() op rhs.Get();                                             \
  }

// op is binary [==, !=, <, <=, >, >=]
// fp_type is [float, double]
#define PROVIDE_MIXED_BFLOAT16T_COMPARISON_OPERATOR(op, fp_type)               \
                                                                               \
  GT_INLINE bool operator op(const bfloat16_t& lhs, const fp_type& rhs)        \
  {                                                                            \
    return static_cast<fp_type>(lhs.Get()) op rhs;                             \
  }                                                                            \
                                                                               \
  GT_INLINE bool operator op(const fp_type& lhs, const bfloat16_t& rhs)        \
  {                                                                            \
    return lhs op static_cast<fp_type>(rhs.Get());                             \
  }

PROVIDE_BFLOAT16T_COMPARISON_OPERATOR(==);
PROVIDE_BFLOAT16T_COMPARISON_OPERATOR(!=);
PROVIDE_BFLOAT16T_COMPARISON_OPERATOR(<);
PROVIDE_BFLOAT16T_COMPARISON_OPERATOR(<=);
PROVIDE_BFLOAT16T_COMPARISON_OPERATOR(>);
PROVIDE_BFLOAT16T_COMPARISON_OPERATOR(>=);

PROVIDE_MIXED_BFLOAT16T_COMPARISON_OPERATOR(==, float);
PROVIDE_MIXED_BFLOAT16T_COMPARISON_OPERATOR(!=, float);
PROVIDE_MIXED_BFLOAT16T_COMPARISON_OPERATOR(<, float);
PROVIDE_MIXED_BFLOAT16T_COMPARISON_OPERATOR(<=, float);
PROVIDE_MIXED_BFLOAT16T_COMPARISON_OPERATOR(>, float);
PROVIDE_MIXED_BFLOAT16T_COMPARISON_OPERATOR(>=, float);

PROVIDE_MIXED_BFLOAT16T_COMPARISON_OPERATOR(==, double);
PROVIDE_MIXED_BFLOAT16T_COMPARISON_OPERATOR(!=, double);
PROVIDE_MIXED_BFLOAT16T_COMPARISON_OPERATOR(<, double);
PROVIDE_MIXED_BFLOAT16T_COMPARISON_OPERATOR(<=, double);
PROVIDE_MIXED_BFLOAT16T_COMPARISON_OPERATOR(>, double);
PROVIDE_MIXED_BFLOAT16T_COMPARISON_OPERATOR(>=, double);

// op is [==, !=]
// int_type is [int]
#define PROVIDE_MIXED_INTEGRAL_BFLOAT16T_COMPARISON_OPERATOR(op, int_type)     \
                                                                               \
  GT_INLINE bool operator op(const bfloat16_t& lhs, const int_type& rhs)       \
  {                                                                            \
    return lhs op static_cast<float>(rhs);                                     \
  }                                                                            \
                                                                               \
  GT_INLINE bool operator op(const int_type& lhs, const bfloat16_t& rhs)       \
  {                                                                            \
    return static_cast<float>(lhs) op rhs;                                     \
  }

PROVIDE_MIXED_INTEGRAL_BFLOAT16T_COMPARISON_OPERATOR(==, int);
PROVIDE_MIXED_INTEGRAL_BFLOAT16T_COMPARISON_OPERATOR(!=, int);

// function is sqrt
GT_INLINE bfloat16_t sqrt(const bfloat16_t& x)
{
#if defined(BFLOAT16T_ON_CUDA_DEVICE)
  return hsqrt(x.Get());
#else
  return std::sqrt(x.Get());
#endif
}

std::ostream& operator<<(std::ostream& s, const bfloat16_t& h)
{
  s << static_cast<float>(h.Get());
  return s;
}

} // namespace gt

#undef GTENSOR_BF16_CUDA_HEADER
#undef BFLOAT16T_ON_CUDA_DEVICE
#undef PROVIDE_BFLOAT16T_UNARY_ARITHMETIC_OPERATOR
#undef PROVIDE_BFLOAT16T_BINARY_ARITHMETIC_OPERATOR
#undef PROVIDE_MIXED_BFLOAT16T_BINARY_ARITHMETIC_OPERATOR
#undef PROVIDE_BFLOAT16T_COMPARISON_OPERATOR
#undef PROVIDE_MIXED_BFLOAT16T_COMPARISON_OPERATOR
#undef PROVIDE_MIXED_INTEGRAL_BFLOAT16T_COMPARISON_OPERATOR

#endif // GTENSOR_BFLOAT16T_H
