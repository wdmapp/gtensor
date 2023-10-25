#ifndef GTENSOR_FLOAT16T_H
#define GTENSOR_FLOAT16T_H

#include <cmath>
#include <iostream>

#if __has_include(<cuda_fp16.h>)
#include <cuda_fp16.h>
#define GTENSOR_FP16_CUDA_HEADER
#elif 0 // TODO check if other fp16 type available, e.g., _Float16
#else
#error "GTENSOR_ENABLE_FP16=ON, but no 16-bit FP type available!"
#endif

namespace gt
{

// ======================================================================
// float16_t

#if defined(GTENSOR_FP16_CUDA_HEADER)
using storage_type = __half;
#else
#error "GTENSOR_ENABLE_FP16=ON, but no 16-bit FP type available!"
#endif

#if defined(GTENSOR_FP16_CUDA_HEADER) && defined(__CUDA_ARCH__) &&             \
  (__CUDA_ARCH__ >= 530)
using compute_type = __half;
#define FLOAT16T_ON_CUDA_DEVICE
#else
using compute_type = float;
#endif

class float16_t
{
public:
  float16_t() = default;
  GT_INLINE float16_t(float x) : x(x){};
  GT_INLINE float16_t(storage_type x) : x(x){};

  GT_INLINE const float16_t& operator=(const float f)
  {
    x = f;
    return *this;
  }
  GT_INLINE compute_type Get() const { return static_cast<compute_type>(x); }

  // update operators [+=, -=, *=, /=]
  GT_INLINE float16_t operator+=(const float16_t& y)
  {
#if defined(FLOAT16T_ON_CUDA_DEVICE)
    x += y.Get();
#else
    x = this->Get() + y.Get();
#endif
    return *this;
  }
  GT_INLINE float16_t operator-=(const float16_t& y)
  {
#if defined(FLOAT16T_ON_CUDA_DEVICE)
    x -= y.Get();
#else
    x = this->Get() - y.Get();
#endif
    return *this;
  }
  GT_INLINE float16_t operator*=(const float16_t& y)
  {
#if defined(FLOAT16T_ON_CUDA_DEVICE)
    x *= y.Get();
#else
    x = this->Get() * y.Get();
#endif
    return *this;
  }
  GT_INLINE float16_t operator/=(const float16_t& y)
  {
#if defined(FLOAT16T_ON_CUDA_DEVICE)
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
#define PROVIDE_FLOAT16T_UNARY_ARITHMETIC_OPERATOR(op)                         \
  GT_INLINE float16_t operator op(const float16_t& rhs)                        \
  {                                                                            \
    return float16_t(op rhs.Get());                                            \
  }

PROVIDE_FLOAT16T_UNARY_ARITHMETIC_OPERATOR(+);
PROVIDE_FLOAT16T_UNARY_ARITHMETIC_OPERATOR(-);

// op is binary [+, -, *, /]
#define PROVIDE_FLOAT16T_BINARY_ARITHMETIC_OPERATOR(op)                        \
  GT_INLINE float16_t operator op(const float16_t& lhs, const float16_t& rhs)  \
  {                                                                            \
    return float16_t(lhs.Get() op rhs.Get());                                  \
  }

PROVIDE_FLOAT16T_BINARY_ARITHMETIC_OPERATOR(+);
PROVIDE_FLOAT16T_BINARY_ARITHMETIC_OPERATOR(-);
PROVIDE_FLOAT16T_BINARY_ARITHMETIC_OPERATOR(*);
PROVIDE_FLOAT16T_BINARY_ARITHMETIC_OPERATOR(/);

// op is binary [+, -, *, /]
// fp_type is [float, double]
#define PROVIDE_MIXED_FLOAT16T_BINARY_ARITHMETIC_OPERATOR(op, fp_type)         \
                                                                               \
  GT_INLINE fp_type operator op(const float16_t& lhs, const fp_type& rhs)      \
  {                                                                            \
    return static_cast<fp_type>(lhs.Get()) op rhs;                             \
  }                                                                            \
                                                                               \
  GT_INLINE fp_type operator op(const fp_type& lhs, const float16_t& rhs)      \
  {                                                                            \
    return lhs op static_cast<fp_type>(rhs.Get());                             \
  }

PROVIDE_MIXED_FLOAT16T_BINARY_ARITHMETIC_OPERATOR(+, float);
PROVIDE_MIXED_FLOAT16T_BINARY_ARITHMETIC_OPERATOR(-, float);
PROVIDE_MIXED_FLOAT16T_BINARY_ARITHMETIC_OPERATOR(*, float);
PROVIDE_MIXED_FLOAT16T_BINARY_ARITHMETIC_OPERATOR(/, float);

PROVIDE_MIXED_FLOAT16T_BINARY_ARITHMETIC_OPERATOR(+, double);
PROVIDE_MIXED_FLOAT16T_BINARY_ARITHMETIC_OPERATOR(-, double);
PROVIDE_MIXED_FLOAT16T_BINARY_ARITHMETIC_OPERATOR(*, double);
PROVIDE_MIXED_FLOAT16T_BINARY_ARITHMETIC_OPERATOR(/, double);

// op is binary [==, !=, <, <=, >, >=]
#define PROVIDE_FLOAT16T_COMPARISON_OPERATOR(op)                               \
  GT_INLINE bool operator op(const float16_t& lhs, const float16_t& rhs)       \
  {                                                                            \
    return lhs.Get() op rhs.Get();                                             \
  }

// op is binary [==, !=, <, <=, >, >=]
// fp_type is [float, double]
#define PROVIDE_MIXED_FLOAT16T_COMPARISON_OPERATOR(op, fp_type)                \
                                                                               \
  GT_INLINE bool operator op(const float16_t& lhs, const fp_type& rhs)         \
  {                                                                            \
    return static_cast<fp_type>(lhs.Get()) op rhs;                             \
  }                                                                            \
                                                                               \
  GT_INLINE bool operator op(const fp_type& lhs, const float16_t& rhs)         \
  {                                                                            \
    return lhs op static_cast<fp_type>(rhs.Get());                             \
  }

PROVIDE_FLOAT16T_COMPARISON_OPERATOR(==);
PROVIDE_FLOAT16T_COMPARISON_OPERATOR(!=);
PROVIDE_FLOAT16T_COMPARISON_OPERATOR(<);
PROVIDE_FLOAT16T_COMPARISON_OPERATOR(<=);
PROVIDE_FLOAT16T_COMPARISON_OPERATOR(>);
PROVIDE_FLOAT16T_COMPARISON_OPERATOR(>=);

PROVIDE_MIXED_FLOAT16T_COMPARISON_OPERATOR(==, float);
PROVIDE_MIXED_FLOAT16T_COMPARISON_OPERATOR(!=, float);
PROVIDE_MIXED_FLOAT16T_COMPARISON_OPERATOR(<, float);
PROVIDE_MIXED_FLOAT16T_COMPARISON_OPERATOR(<=, float);
PROVIDE_MIXED_FLOAT16T_COMPARISON_OPERATOR(>, float);
PROVIDE_MIXED_FLOAT16T_COMPARISON_OPERATOR(>=, float);

PROVIDE_MIXED_FLOAT16T_COMPARISON_OPERATOR(==, double);
PROVIDE_MIXED_FLOAT16T_COMPARISON_OPERATOR(!=, double);
PROVIDE_MIXED_FLOAT16T_COMPARISON_OPERATOR(<, double);
PROVIDE_MIXED_FLOAT16T_COMPARISON_OPERATOR(<=, double);
PROVIDE_MIXED_FLOAT16T_COMPARISON_OPERATOR(>, double);
PROVIDE_MIXED_FLOAT16T_COMPARISON_OPERATOR(>=, double);

// op is [==, !=]
// int_type is [int]
#define PROVIDE_MIXED_INTEGRAL_FLOAT16T_COMPARISON_OPERATOR(op, int_type)      \
                                                                               \
  GT_INLINE bool operator op(const float16_t& lhs, const int_type& rhs)        \
  {                                                                            \
    return lhs op static_cast<float>(rhs);                                     \
  }                                                                            \
                                                                               \
  GT_INLINE bool operator op(const int_type& lhs, const float16_t& rhs)        \
  {                                                                            \
    return static_cast<float>(lhs) op rhs;                                     \
  }

PROVIDE_MIXED_INTEGRAL_FLOAT16T_COMPARISON_OPERATOR(==, int);
PROVIDE_MIXED_INTEGRAL_FLOAT16T_COMPARISON_OPERATOR(!=, int);

// function is sqrt
GT_INLINE float16_t sqrt(const float16_t& x)
{
#if defined(FLOAT16T_ON_CUDA_DEVICE)
  return hsqrt(x.Get());
#else
  return std::sqrt(x.Get());
#endif
}

std::ostream& operator<<(std::ostream& s, const float16_t& h)
{
  s << static_cast<float>(h.Get());
  return s;
}

} // namespace gt

#undef PROVIDE_FLOAT16T_UNARY_ARITHMETIC_OPERATOR
#undef PROVIDE_FLOAT16T_BINARY_ARITHMETIC_OPERATOR
#undef PROVIDE_MIXED_FLOAT16T_BINARY_ARITHMETIC_OPERATOR
#undef PROVIDE_FLOAT16T_COMPARISON_OPERATOR
#undef PROVIDE_MIXED_FLOAT16T_COMPARISON_OPERATOR
#undef PROVIDE_MIXED_INTEGRAL_FLOAT16T_COMPARISON_OPERATOR

#endif
