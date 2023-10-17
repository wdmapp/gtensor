#ifndef GTENSOR_COMPLEX_FLOAT16T_H
#define GTENSOR_COMPLEX_FLOAT16T_H

#include "complex.h"
#include "float16_t.h"
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
// complex_float16_t

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

template<class T> class complex;

template<> class complex<float>;
template<> class complex<double>;
template<> class complex<long double>;

// operators:
template<class T> constexpr complex<T> operator+(
    const complex<T>&, const complex<T>&);
template<class T> constexpr complex<T> operator+(const complex<T>&, const T&);
template<class T> constexpr complex<T> operator+(const T&, const complex<T>&);

template<class T> constexpr complex<T> operator-(
    const complex<T>&, const complex<T>&);
template<class T> constexpr complex<T> operator-(const complex<T>&, const T&);
template<class T> constexpr complex<T> operator-(const T&, const complex<T>&);

template<class T> constexpr complex<T> operator*(
    const complex<T>&, const complex<T>&);
template<class T> constexpr complex<T> operator*(const complex<T>&, const T&);
template<class T> constexpr complex<T> operator*(const T&, const complex<T>&);

template<class T> constexpr complex<T> operator/(
    const complex<T>&, const complex<T>&);
template<class T> constexpr complex<T> operator/(const complex<T>&, const T&);
template<class T> constexpr complex<T> operator/(const T&, const complex<T>&);

template<class T> constexpr complex<T> operator+(const complex<T>&);
template<class T> constexpr complex<T> operator-(const complex<T>&);

template<class T> constexpr bool operator==(const complex<T>&, const complex<T>&);
template<class T> constexpr bool operator==(const complex<T>&, const T&);
template<class T> constexpr bool operator==(const T&, const complex<T>&);

template<class T> constexpr bool operator!=(const complex<T>&, const complex<T>&);
template<class T> constexpr bool operator!=(const complex<T>&, const T&);
template<class T> constexpr bool operator!=(const T&, const complex<T>&);

template<class T, class CharT, class Traits>
basic_istream<CharT, Traits>&
operator>>(basic_istream<CharT, Traits>&, complex<T>&);

template<class T, class CharT, class Traits>
basic_ostream<CharT, Traits>&
operator<<(basic_ostream<CharT, Traits>&, const complex<T>&);

// values:
template<class T> constexpr T real(const complex<T>&);
template<class T> constexpr T imag(const complex<T>&);

template<class T> T abs(const complex<T>&);
template<class T> T arg(const complex<T>&);
template<class T> constexpr T norm(const complex<T>&);

template<class T> constexpr complex<T> conj(const complex<T>&);
template<class T> complex<T> proj(const complex<T>&);
template<class T> complex<T> polar(const T&, const T& = 0);

// transcendentals:
template<class T> complex<T> acos(const complex<T>&);
template<class T> complex<T> asin(const complex<T>&);
template<class T> complex<T> atan(const complex<T>&);

template<class T> complex<T> acosh(const complex<T>&);
template<class T> complex<T> asinh(const complex<T>&);
template<class T> complex<T> atanh(const complex<T>&);

template<class T> complex<T> cos  (const complex<T>&);
template<class T> complex<T> cosh (const complex<T>&);
template<class T> complex<T> exp  (const complex<T>&);
template<class T> complex<T> log  (const complex<T>&);
template<class T> complex<T> log10(const complex<T>&);

template<class T> complex<T> pow(const complex<T>&, const T&);
template<class T> complex<T> pow(const complex<T>&, const complex<T>&);
template<class T> complex<T> pow(const T&, const complex<T>&);

template<class T> complex<T> sin (const complex<T>&);
template<class T> complex<T> sinh(const complex<T>&);
template<class T> complex<T> sqrt(const complex<T>&);
template<class T> complex<T> tan (const complex<T>&);
template<class T> complex<T> tanh(const complex<T>&);

// complex literals:
inline namespace literals {
    inline namespace complex_literals {
        constexpr complex<long double> operator""il(long double);
        constexpr complex<long double> operator""il(unsigned long long);
        constexpr complex<double> operator""i(long double);
        constexpr complex<double> operator""i(unsigned long long);
        constexpr complex<float> operator""if(long double);
        constexpr complex<float> operator""if(unsigned long long);
    }
}

template<class T>
class complex {
public:
    typedef T value_type;
    constexpr complex(const T& re = T(), const T& im = T());
    constexpr complex(const complex&) = default;
    template<class X> constexpr explicit(/* see constructor page */)
        complex(const complex<X>&);
 
    constexpr T real() const;
    constexpr void real(T);
    constexpr T imag() const;
    constexpr void imag(T);
 
    constexpr complex<T>& operator= (const T&);
    constexpr complex<T>& operator+=(const T&);
    constexpr complex<T>& operator-=(const T&);
    constexpr complex<T>& operator*=(const T&);
    constexpr complex<T>& operator/=(const T&);
 
    constexpr complex& operator=(const complex&);
    template<class X> constexpr complex<T>& operator= (const complex<X>&);
    template<class X> constexpr complex<T>& operator+=(const complex<X>&);
    template<class X> constexpr complex<T>& operator-=(const complex<X>&);
    template<class X> constexpr complex<T>& operator*=(const complex<X>&);
    template<class X> constexpr complex<T>& operator/=(const complex<X>&);
};

} // namespace gt

#undef PROVIDE_COMPLEX_FLOAT16T_BINARY_ARITHMETIC_OPERATOR

#undef FLOAT16T_ON_CUDA_DEVICE

#endif // GTENSOR_COMPLEX_FLOAT16T_H
