// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Adapted from the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _SYCL_EXT_CPLX_COMPLEX
#define _SYCL_EXT_CPLX_COMPLEX

// clang-format off

// Last time synced SyclCPLX with llvm-project
// https://github.com/llvm/llvm-project/commit/385cc25a531a72c393cee44689e2c3194615bcec

/*
    complex synopsis

namespace sycl::ext::cplx
{

template<class T>
class complex
{
public:
    typedef T value_type;

    complex(const T& re = T(), const T& im = T()); // constexpr in C++14
    complex(const complex&);  // constexpr in C++14
    template<class X> complex(const complex<X>&);  // constexpr in C++14

    T real() const; // constexpr in C++14
    T imag() const; // constexpr in C++14

    void real(T);
    void imag(T);

    complex<T>& operator= (const T&);
    complex<T>& operator+=(const T&);
    complex<T>& operator-=(const T&);
    complex<T>& operator*=(const T&);
    complex<T>& operator/=(const T&);

    complex& operator=(const complex&);
    template<class X> complex<T>& operator= (const complex<X>&);
    template<class X> complex<T>& operator+=(const complex<X>&);
    template<class X> complex<T>& operator-=(const complex<X>&);
    template<class X> complex<T>& operator*=(const complex<X>&);
    template<class X> complex<T>& operator/=(const complex<X>&);
};

template<>
class complex<sycl::half>
{
public:
    typedef sycl::half value_type;

    constexpr complex(sycl::half re = 0.0f, sycl::half im = 0.0f);
    explicit constexpr complex(const complex<float>&);
    explicit constexpr complex(const complex<double>&);

    constexpr operator std::complex<sycl::half>();

    constexpr sycl::half real() const;
    void real(sycl::half);
    constexpr sycl::half imag() const;
    void imag(sycl::half);

    complex<sycl::half>& operator= (sycl::half);
    complex<sycl::half>& operator+=(sycl::half);
    complex<sycl::half>& operator-=(sycl::half);
    complex<sycl::half>& operator*=(sycl::half);
    complex<sycl::half>& operator/=(sycl::half);

    complex<sycl::half>& operator=(const complex<sycl::half>&);
    template<class X> complex<sycl::half>& operator= (const complex<X>&);
    template<class X> complex<sycl::half>& operator+=(const complex<X>&);
    template<class X> complex<sycl::half>& operator-=(const complex<X>&);
    template<class X> complex<sycl::half>& operator*=(const complex<X>&);
    template<class X> complex<sycl::half>& operator/=(const complex<X>&);
};

template<>
class complex<float>
{
public:
    typedef float value_type;

    constexpr complex(float re = 0.0f, float im = 0.0f);
    constexpr complex(const complex<sycl::half>&);
    explicit constexpr complex(const complex<double>&);

    constexpr complex(const std::complex<float>&);
    constexpr operator std::complex<float>();

    constexpr float real() const;
    void real(float);
    constexpr float imag() const;
    void imag(float);

    complex<float>& operator= (float);
    complex<float>& operator+=(float);
    complex<float>& operator-=(float);
    complex<float>& operator*=(float);
    complex<float>& operator/=(float);

    complex<float>& operator=(const complex<float>&);
    template<class X> complex<float>& operator= (const complex<X>&);
    template<class X> complex<float>& operator+=(const complex<X>&);
    template<class X> complex<float>& operator-=(const complex<X>&);
    template<class X> complex<float>& operator*=(const complex<X>&);
    template<class X> complex<float>& operator/=(const complex<X>&);
};

template<>
class complex<double>
{
public:
    typedef double value_type;

    constexpr complex(double re = 0.0, double im = 0.0);
    constexpr complex(const complex<sycl::half>&);
    constexpr complex(const complex<float>&);

    constexpr complex(const std::complex<double>&);
    constexpr operator std::complex<double>();

    constexpr double real() const;
    void real(double);
    constexpr double imag() const;
    void imag(double);

    complex<double>& operator= (double);
    complex<double>& operator+=(double);
    complex<double>& operator-=(double);
    complex<double>& operator*=(double);
    complex<double>& operator/=(double);
    complex<double>& operator=(const complex<double>&);

    template<class X> complex<double>& operator= (const complex<X>&);
    template<class X> complex<double>& operator+=(const complex<X>&);
    template<class X> complex<double>& operator-=(const complex<X>&);
    template<class X> complex<double>& operator*=(const complex<X>&);
    template<class X> complex<double>& operator/=(const complex<X>&);
};


// 26.3.6 operators:
template<class T> complex<T> operator+(const complex<T>&, const complex<T>&);
template<class T> complex<T> operator+(const complex<T>&, const T&);
template<class T> complex<T> operator+(const T&, const complex<T>&);
template<class T> complex<T> operator-(const complex<T>&, const complex<T>&);
template<class T> complex<T> operator-(const complex<T>&, const T&);
template<class T> complex<T> operator-(const T&, const complex<T>&);
template<class T> complex<T> operator*(const complex<T>&, const complex<T>&);
template<class T> complex<T> operator*(const complex<T>&, const T&);
template<class T> complex<T> operator*(const T&, const complex<T>&);
template<class T> complex<T> operator/(const complex<T>&, const complex<T>&);
template<class T> complex<T> operator/(const complex<T>&, const T&);
template<class T> complex<T> operator/(const T&, const complex<T>&);
template<class T> complex<T> operator+(const complex<T>&);
template<class T> complex<T> operator-(const complex<T>&);
template<class T> bool operator==(const complex<T>&, const complex<T>&); // constexpr in C++14
template<class T> bool operator==(const complex<T>&, const T&); // constexpr in C++14
template<class T> bool operator==(const T&, const complex<T>&); // constexpr in C++14
template<class T> bool operator!=(const complex<T>&, const complex<T>&); // constexpr in C++14
template<class T> bool operator!=(const complex<T>&, const T&); // constexpr in C++14
template<class T> bool operator!=(const T&, const complex<T>&); // constexpr in C++14

template<class T, class charT, class traits>
  basic_istream<charT, traits>&
  operator>>(basic_istream<charT, traits>&, complex<T>&);
template<class T, class charT, class traits>
  basic_ostream<charT, traits>&
  operator<<(basic_ostream<charT, traits>&, const complex<T>&);
template<class T>
  const sycl::stream&
  operator<<(const sycl::stream&, const complex<T>&);

// 26.3.7 values:

template<class T>              T real(const complex<T>&); // constexpr in C++14
                          double real(double);            // constexpr in C++14
template<Integral T>      double real(T);                 // constexpr in C++14
                          float  real(float);             // constexpr in C++14

template<class T>              T imag(const complex<T>&); // constexpr in C++14
                          double imag(double);            // constexpr in C++14
template<Integral T>      double imag(T);                 // constexpr in C++14
                          float  imag(float);             // constexpr in C++14

template<class T> T abs(const complex<T>&);

template<class T>              T arg(const complex<T>&);
                          double arg(double);
template<Integral T>      double arg(T);
                          float  arg(float);

template<class T>              T norm(const complex<T>&);
                          double norm(double);
template<Integral T>      double norm(T);
                          float  norm(float);

template<class T>      complex<T>           conj(const complex<T>&);
                       complex<double>      conj(double);
template<Integral T>   complex<double>      conj(T);
                       complex<float>       conj(float);

template<class T>    complex<T>           proj(const complex<T>&);
                     complex<double>      proj(double);
template<Integral T> complex<double>      proj(T);
                     complex<float>       proj(float);

template<class T> complex<T> polar(const T&, const T& = T());

// 26.3.8 transcendentals:
template<class T> complex<T> acos(const complex<T>&);
template<class T> complex<T> asin(const complex<T>&);
template<class T> complex<T> atan(const complex<T>&);
template<class T> complex<T> acosh(const complex<T>&);
template<class T> complex<T> asinh(const complex<T>&);
template<class T> complex<T> atanh(const complex<T>&);
template<class T> complex<T> cos (const complex<T>&);
template<class T> complex<T> cosh (const complex<T>&);
template<class T> complex<T> exp (const complex<T>&);
template<class T> complex<T> log (const complex<T>&);
template<class T> complex<T> log10(const complex<T>&);

template<class T> complex<T> pow(const complex<T>&, const T&);
template<class T> complex<T> pow(const complex<T>&, const complex<T>&);
template<class T> complex<T> pow(const T&, const complex<T>&);

template<class T> complex<T> sin (const complex<T>&);
template<class T> complex<T> sinh (const complex<T>&);
template<class T> complex<T> sqrt (const complex<T>&);
template<class T> complex<T> tan (const complex<T>&);
template<class T> complex<T> tanh (const complex<T>&);

}  // sycl::ext::cplx

*/

// clang-format on

#ifndef _SYCL_CPLX_NAMESPACE
#ifdef __HIPSYCL__
#define _SYCL_CPLX_NAMESPACE hipsycl::sycl::ext::cplx
#else
#define _SYCL_CPLX_NAMESPACE sycl::ext::cplx
#endif
#endif

#define _SYCL_EXT_CPLX_BEGIN_NAMESPACE_STD namespace _SYCL_CPLX_NAMESPACE {
#define _SYCL_EXT_CPLX_END_NAMESPACE_STD }
#define _SYCL_EXT_CPLX_INLINE_VISIBILITY                                       \
  [[gnu::always_inline]] [[clang::always_inline]] inline

#include <complex>
#include <sstream> // for std::basic_ostringstream
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
// support oneAPI 2022.X and earlier
#include <CL/sycl.hpp>
#else
#error "SYCL header not found"
#endif
#include <type_traits>

_SYCL_EXT_CPLX_BEGIN_NAMESPACE_STD

namespace cplex::detail {
template <class _Tp> struct __numeric_type {
  static void __test(...);
  static sycl::half __test(sycl::half);
  static float __test(float);
  static double __test(char);
  static double __test(int);
  static double __test(unsigned);
  static double __test(long);
  static double __test(unsigned long);
  static double __test(long long);
  static double __test(unsigned long long);
  static double __test(double);

  typedef decltype(__test(std::declval<_Tp>())) type;
  static const bool value = !std::is_same<type, void>::value;
};

template <> struct __numeric_type<void> { static const bool value = true; };

template <class _A1, class _A2 = void, class _A3 = void,
          bool = __numeric_type<_A1>::value &&__numeric_type<_A2>::value
              &&__numeric_type<_A3>::value>
class __promote_imp {
public:
  static const bool value = false;
};

template <class _A1, class _A2, class _A3>
class __promote_imp<_A1, _A2, _A3, true> {
private:
  typedef typename __promote_imp<_A1>::type __type1;
  typedef typename __promote_imp<_A2>::type __type2;
  typedef typename __promote_imp<_A3>::type __type3;

public:
  typedef decltype(__type1() + __type2() + __type3()) type;
  static const bool value = true;
};

template <class _A1, class _A2> class __promote_imp<_A1, _A2, void, true> {
private:
  typedef typename __promote_imp<_A1>::type __type1;
  typedef typename __promote_imp<_A2>::type __type2;

public:
  typedef decltype(__type1() + __type2()) type;
  static const bool value = true;
};

template <class _A1> class __promote_imp<_A1, void, void, true> {
public:
  typedef typename __numeric_type<_A1>::type type;
  static const bool value = true;
};

template <class _A1, class _A2 = void, class _A3 = void>
class __promote : public __promote_imp<_A1, _A2, _A3> {};
}

template <class _Tp, class = void> class complex;

template <class _Tp>
struct is_gencomplex
    : std::integral_constant<bool,
                             std::is_same_v<_Tp, complex<double>> ||
                                 std::is_same_v<_Tp, complex<float>> ||
                                 std::is_same_v<_Tp, complex<sycl::half>>> {};

template <class _Tp>
struct is_genfloat
    : std::integral_constant<bool, std::is_same_v<_Tp, double> ||
                                       std::is_same_v<_Tp, float> ||
                                       std::is_same_v<_Tp, sycl::half>> {};

template <class _Tp>
class complex<_Tp, typename std::enable_if<is_genfloat<_Tp>::value>::type> {
public:
  typedef _Tp value_type;

private:
  value_type __re_;
  value_type __im_;

public:
  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr complex(value_type __re = value_type(), value_type __im = value_type()) : __re_(__re), __im_(__im) {

  }

  template <typename _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr complex(const complex<_Xp> &__c) : __re_(__c.real()), __im_(__c.imag()) {

  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr complex(const std::complex<value_type> &__c) : __re_(__c.real()), __im_(__c.imag()) {

  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr operator std::complex<value_type>() const {
    return std::complex<value_type>(__re_, __im_);
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr value_type real() const {
    return __re_;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr value_type imag() const {
    return __im_;
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY void real(value_type __re) { __re_ = __re; }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY void imag(value_type __im) { __im_ = __im; }

  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Xp> &operator=(value_type __re) {
    __re_ = __re;
    __im_ = value_type();
    return *this;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex
  &operator+=(complex<value_type> &__c, value_type __re) {
    __c.__re_ += __re;
    return __c;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex
  &operator-=(complex<value_type> &__c, value_type __re) {
    __c.__re_ -= __re;
    return __c;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex
  &operator*=(complex<value_type> &__c, value_type __re) {
    __c.__re_ *= __re;
    __c.__im_ *= __re;
    return __c;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex
  &operator/=(complex<value_type> &__c, value_type __re) {
    __c.__re_ /= __re;
    __c.__im_ /= __re;
    return __c;
  }

  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Xp> &operator=(const complex<_Xp> &__c) {
    __re_ = __c.real();
    __im_ = __c.imag();
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex
  &operator+=(complex<value_type> &__x, const complex<_Xp> &__y) {
    __x.__re_ += __y.real();
    __x.__im_ += __y.imag();
    return __x;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex
  &operator-=(complex<value_type> &__x, const complex<_Xp> &__y) {
    __x.__re_ -= __y.real();
    __x.__im_ -= __y.imag();
    return __x;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex
  &operator*=(complex<value_type> &__x, const complex<_Xp> &__y) {
    __x = __x * complex(__y.real(), __y.imag());
    return __x;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex
  &operator/=(complex<value_type> &__x, const complex<_Xp> &__y) {
    __x = __x / complex(__y.real(), __y.imag());
    return __x;
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex<value_type>
  operator+(const complex<value_type> &__x, const complex<value_type> &__y) {
    complex<value_type> __t(__x);
    __t += __y;
    return __t;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex<value_type>
  operator+(const complex<value_type> &__x, value_type __y) {
    complex<value_type> __t(__x);
    __t += __y;
    return __t;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex<value_type>
  operator+(value_type __x, const complex<value_type> &__y) {
    complex<value_type> __t(__y);
    __t += __x;
    return __t;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex<value_type>
  operator+(const complex<value_type> &__x) {
    return __x;
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex<value_type>
  operator-(const complex<value_type> &__x, const complex<value_type> &__y) {
    complex<value_type> __t(__x);
    __t -= __y;
    return __t;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex<value_type>
  operator-(const complex<value_type> &__x, value_type __y) {
    complex<value_type> __t(__x);
    __t -= __y;
    return __t;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex<value_type>
  operator-(value_type __x, const complex<value_type> &__y) {
    complex<value_type> __t(-__y);
    __t += __x;
    return __t;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex<value_type>
  operator-(const complex<value_type> &__x) {
    return complex<value_type>(-__x.__re_, -__x.__im_);
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex<value_type>
  operator*(const complex<value_type> &__z, const complex<value_type> &__w) {
    value_type __a = __z.__re_;
    value_type __b = __z.__im_;
    value_type __c = __w.__re_;
    value_type __d = __w.__im_;
    value_type __ac = __a * __c;
    value_type __bd = __b * __d;
    value_type __ad = __a * __d;
    value_type __bc = __b * __c;
    value_type __x = __ac - __bd;
    value_type __y = __ad + __bc;
    if (sycl::isnan(__x) && sycl::isnan(__y)) {
      bool __recalc = false;
      if (sycl::isinf(__a) || sycl::isinf(__b)) {
        __a = sycl::copysign(sycl::isinf(__a) ? value_type(1) : value_type(0), __a);
        __b = sycl::copysign(sycl::isinf(__b) ? value_type(1) : value_type(0), __b);
        if (sycl::isnan(__c))
          __c = sycl::copysign(value_type(0), __c);
        if (sycl::isnan(__d))
          __d = sycl::copysign(value_type(0), __d);
        __recalc = true;
      }
      if (sycl::isinf(__c) || sycl::isinf(__d)) {
        __c = sycl::copysign(sycl::isinf(__c) ? value_type(1) : value_type(0), __c);
        __d = sycl::copysign(sycl::isinf(__d) ? value_type(1) : value_type(0), __d);
        if (sycl::isnan(__a))
          __a = sycl::copysign(value_type(0), __a);
        if (sycl::isnan(__b))
          __b = sycl::copysign(value_type(0), __b);
        __recalc = true;
      }
      if (!__recalc && (sycl::isinf(__ac) || sycl::isinf(__bd) ||
                        sycl::isinf(__ad) || sycl::isinf(__bc))) {
        if (sycl::isnan(__a))
          __a = sycl::copysign(value_type(0), __a);
        if (sycl::isnan(__b))
          __b = sycl::copysign(value_type(0), __b);
        if (sycl::isnan(__c))
          __c = sycl::copysign(value_type(0), __c);
        if (sycl::isnan(__d))
          __d = sycl::copysign(value_type(0), __d);
        __recalc = true;
      }
      if (__recalc) {
        __x = value_type(INFINITY) * (__a * __c - __b * __d);
        __y = value_type(INFINITY) * (__a * __d + __b * __c);
      }
    }
    return complex<value_type>(__x, __y);
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex<value_type>
  operator*(const complex<value_type> &__x, value_type __y) {
    complex<value_type> __t(__x);
    __t *= __y;
    return __t;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex<value_type>
  operator*(value_type __x, const complex<value_type> &__y) {
    complex<value_type> __t(__y);
    __t *= __x;
    return __t;
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex<value_type>
  operator/(const complex<value_type> &__z, const complex<value_type> &__w) {
    int __ilogbw = 0;
    value_type __a = __z.__re_;
    value_type __b = __z.__im_;
    value_type __c = __w.__re_;
    value_type __d = __w.__im_;
    value_type __logbw = sycl::logb(sycl::fmax(sycl::fabs(__c), sycl::fabs(__d)));
    if (sycl::isfinite(__logbw)) {
      __ilogbw = static_cast<int>(__logbw);
      __c = sycl::ldexp(__c, -__ilogbw);
      __d = sycl::ldexp(__d, -__ilogbw);
    }
    value_type __denom = __c * __c + __d * __d;
    value_type __x = sycl::ldexp((__a * __c + __b * __d) / __denom, -__ilogbw);
    value_type __y = sycl::ldexp((__b * __c - __a * __d) / __denom, -__ilogbw);
    if (sycl::isnan(__x) && sycl::isnan(__y)) {
      if ((__denom == value_type(0)) && (!sycl::isnan(__a) || !sycl::isnan(__b))) {
        __x = sycl::copysign(value_type(INFINITY), __c) * __a;
        __y = sycl::copysign(value_type(INFINITY), __c) * __b;
      } else if ((sycl::isinf(__a) || sycl::isinf(__b)) && sycl::isfinite(__c) &&
                 sycl::isfinite(__d)) {
        __a = sycl::copysign(sycl::isinf(__a) ? value_type(1) : value_type(0), __a);
        __b = sycl::copysign(sycl::isinf(__b) ? value_type(1) : value_type(0), __b);
        __x = value_type(INFINITY) * (__a * __c + __b * __d);
        __y = value_type(INFINITY) * (__b * __c - __a * __d);
      } else if (sycl::isinf(__logbw) && __logbw > value_type(0) &&
                 sycl::isfinite(__a) && sycl::isfinite(__b)) {
        __c = sycl::copysign(sycl::isinf(__c) ? value_type(1) : value_type(0), __c);
        __d = sycl::copysign(sycl::isinf(__d) ? value_type(1) : value_type(0), __d);
        __x = value_type(0) * (__a * __c + __b * __d);
        __y = value_type(0) * (__b * __c - __a * __d);
      }
    }
    return complex<value_type>(__x, __y);
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex<value_type>
  operator/(const complex<value_type> &__x, value_type __y) {
    return complex<value_type>(__x.__re_ / __y, __x.__im_ / __y);
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex<value_type>
  operator/(value_type __x, const complex<value_type> &__y) {
    complex<value_type> __t(__x);
    __t /= __y;
    return __t;
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend constexpr bool
  operator==(const complex<value_type> &__x, const complex<value_type> &__y) {
    return __x.__re_ == __y.__re_ && __x.__im_ == __y.__im_;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend constexpr bool
  operator==(const complex<value_type> &__x, value_type __y) {
    return __x.__re_ == __y && __x.__im_ == 0;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend constexpr bool
  operator==(value_type __x, const complex<value_type> &__y) {
    return __x == __y.__re_ && 0 == __y.__im_;
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend constexpr bool
  operator!=(const complex<value_type> &__x, const complex<value_type> &__y) {
    return !(__x == __y);
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend constexpr bool
  operator!=(const complex<value_type> &__x, value_type __y) {
    return !(__x == __y);
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend constexpr bool
  operator!=(value_type __x, const complex<value_type> &__y) {
    return !(__x == __y);
  }

  template <class _CharT, class _Traits>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend std::basic_istream<_CharT, _Traits>
  &operator>>(std::basic_istream<_CharT, _Traits> &__is, complex<value_type> &__x) {
    if (__is.good()) {
      ws(__is);
      if (__is.peek() == _CharT('(')) {
        __is.get();
        value_type __r;
        __is >> __r;
        if (!__is.fail()) {
          ws(__is);
          _CharT __c = __is.peek();
          if (__c == _CharT(',')) {
            __is.get();
            value_type __i;
            __is >> __i;
            if (!__is.fail()) {
              ws(__is);
              __c = __is.peek();
              if (__c == _CharT(')')) {
                __is.get();
                __x = complex<value_type>(__r, __i);
              } else
                __is.setstate(__is.failbit);
            } else
              __is.setstate(__is.failbit);
          } else if (__c == _CharT(')')) {
            __is.get();
            __x = complex<value_type>(__r, value_type(0));
          } else
            __is.setstate(__is.failbit);
        } else
          __is.setstate(__is.failbit);
      } else {
        value_type __r;
        __is >> __r;
        if (!__is.fail())
          __x = complex<value_type>(__r, value_type(0));
        else
          __is.setstate(__is.failbit);
      }
    } else
      __is.setstate(__is.failbit);
    return __is;
  }

  template <class _CharT, class _Traits>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend std::basic_ostream<_CharT, _Traits>
  &operator<<(std::basic_ostream<_CharT, _Traits> &__os, const complex<value_type> &__x) {
    std::basic_ostringstream<_CharT, _Traits> __s;
    __s.flags(__os.flags());
    __s.imbue(__os.getloc());
    __s.precision(__os.precision());
    __s << '(' << __x.__re_ << ',' << __x.__im_ << ')';
    return __os << __s.str();
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend const sycl::stream
  &operator<<(const sycl::stream &__ss, const complex<value_type> &_x) {
    return __ss << "(" << _x.__re_ << "," << _x.__im_ << ")";
  }
};

namespace cplex::detail {
template <class _Tp, bool = std::is_integral<_Tp>::value,
          bool = is_genfloat<_Tp>::value>
struct __libcpp_complex_overload_traits {};

// Integral Types
template <class _Tp> struct __libcpp_complex_overload_traits<_Tp, true, false> {
  typedef double _ValueType;
  typedef complex<double> _ComplexType;
};

// Floating point types
template <class _Tp> struct __libcpp_complex_overload_traits<_Tp, false, true> {
  typedef _Tp _ValueType;
  typedef complex<_Tp> _ComplexType;
};
}

// real

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr _Tp real(const complex<_Tp> &__c) {
  return __c.real();
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr
    typename cplex::detail::__libcpp_complex_overload_traits<_Tp>::_ValueType
    real(_Tp __re) {
  return __re;
}

// imag

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr _Tp imag(const complex<_Tp> &__c) {
  return __c.imag();
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr
    typename cplex::detail::__libcpp_complex_overload_traits<_Tp>::_ValueType
    imag(_Tp) {
  return 0;
}

// abs

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY _Tp abs(const complex<_Tp> &__c) {
  return sycl::hypot(__c.real(), __c.imag());
}

// arg

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY _Tp arg(const complex<_Tp> &__c) {
  return sycl::atan2(__c.imag(), __c.real());
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY
typename cplex::detail::__libcpp_complex_overload_traits<_Tp>::_ValueType arg(_Tp __re) {
  typedef typename cplex::detail::__libcpp_complex_overload_traits<_Tp>::_ValueType _ValueType;
  return sycl::atan2<_ValueType>(0, __re);
}

// norm

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY _Tp norm(const complex<_Tp> &__c) {
  if (sycl::isinf(__c.real()))
    return sycl::fabs(__c.real());
  if (sycl::isinf(__c.imag()))
    return sycl::fabs(__c.imag());
  return __c.real() * __c.real() + __c.imag() * __c.imag();
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename cplex::detail::__libcpp_complex_overload_traits<_Tp>::_ValueType
    norm(_Tp __re) {
  typedef typename cplex::detail::__libcpp_complex_overload_traits<_Tp>::_ValueType _ValueType;
  return static_cast<_ValueType>(__re) * __re;
}

// conj

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp> conj(const complex<_Tp> &__c) {
  return complex<_Tp>(__c.real(), -__c.imag());
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename cplex::detail::__libcpp_complex_overload_traits<_Tp>::_ComplexType
    conj(_Tp __re) {
  typedef
      typename cplex::detail::__libcpp_complex_overload_traits<_Tp>::_ComplexType _ComplexType;
  return _ComplexType(__re);
}

// proj

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp> proj(const complex<_Tp> &__c) {
  complex<_Tp> __r = __c;
  if (sycl::isinf(__c.real()) || sycl::isinf(__c.imag()))
    __r = complex<_Tp>(INFINITY, sycl::copysign(_Tp(0), __c.imag()));
  return __r;
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY
  typename cplex::detail::__libcpp_complex_overload_traits<_Tp>::_ComplexType
  proj(_Tp __re) {
  typedef typename cplex::detail::__libcpp_complex_overload_traits<_Tp>::_ComplexType _ComplexType;

  if constexpr(!std::is_integral_v<_Tp>) {
    if (sycl::isinf(__re))
      __re = sycl::fabs(__re);
  }

  return _ComplexType(__re);
}

// polar

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp> polar(const _Tp &__rho, const _Tp &__theta = _Tp()) {
  if (sycl::isnan(__rho) || sycl::signbit(__rho))
    return complex<_Tp>(_Tp(NAN), _Tp(NAN));
  if (sycl::isnan(__theta)) {
    if (sycl::isinf(__rho))
      return complex<_Tp>(__rho, __theta);
    return complex<_Tp>(__theta, __theta);
  }
  if (sycl::isinf(__theta)) {
    if (sycl::isinf(__rho))
      return complex<_Tp>(__rho, _Tp(NAN));
    return complex<_Tp>(_Tp(NAN), _Tp(NAN));
  }
  _Tp __x = __rho * sycl::cos(__theta);
  if (sycl::isnan(__x))
    __x = 0;
  _Tp __y = __rho * sycl::sin(__theta);
  if (sycl::isnan(__y))
    __y = 0;
  return complex<_Tp>(__x, __y);
}

// log

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp> log(const complex<_Tp> &__x) {
  return complex<_Tp>(sycl::log(abs(__x)), arg(__x));
}

// log10

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp> log10(const complex<_Tp> &__x) {
  return log(__x) / sycl::log(_Tp(10));
}

// sqrt

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp> sqrt(const complex<_Tp> &__x) {
  if (sycl::isinf(__x.imag()))
    return complex<_Tp>(_Tp(INFINITY), __x.imag());
  if (sycl::isinf(__x.real())) {
    if (__x.real() > _Tp(0))
      return complex<_Tp>(__x.real(), sycl::isnan(__x.imag())
                                          ? __x.imag()
                                          : sycl::copysign(_Tp(0), __x.imag()));
    return complex<_Tp>(sycl::isnan(__x.imag()) ? __x.imag() : _Tp(0),
                        sycl::copysign(__x.real(), __x.imag()));
  }
  return polar(sycl::sqrt(abs(__x)), arg(__x) / _Tp(2));
}

// exp

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp> exp(const complex<_Tp> &__x) {
  _Tp __i = __x.imag();
  if (__i == 0) {
    return complex<_Tp>(sycl::exp(__x.real()),
                        sycl::copysign(_Tp(0), __x.imag()));
  }
  if (sycl::isinf(__x.real())) {
    if (__x.real() < _Tp(0)) {
      if (!sycl::isfinite(__i))
        __i = _Tp(1);
    } else if (__i == 0 || !sycl::isfinite(__i)) {
      if (sycl::isinf(__i))
        __i = _Tp(NAN);
      return complex<_Tp>(__x.real(), __i);
    }
  }
  _Tp __e = sycl::exp(__x.real());
  return complex<_Tp>(__e * sycl::cos(__i), __e * sycl::sin(__i));
}

// pow

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp> pow(const complex<_Tp> &__x,
                                                  const complex<_Tp> &__y) {
  return exp(__y * log(__x));
}

template <class _Tp, class _Up,
          class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<typename cplex::detail::__promote<_Tp, _Up>::type>
pow(const complex<_Tp> &__x, const complex<_Up> &__y) {
  typedef complex<typename cplex::detail::__promote<_Tp, _Up>::type> result_type;
  return pow(result_type(__x), result_type(__y));
}

template <class _Tp, class _Up,
          class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename std::enable_if<is_genfloat<_Up>::value,
                            complex<typename cplex::detail::__promote<_Tp, _Up>::type>>::type
    pow(const complex<_Tp> &__x, const _Up &__y) {
  typedef complex<typename cplex::detail::__promote<_Tp, _Up>::type> result_type;
  return pow(result_type(__x), result_type(__y));
}

template <class _Tp, class _Up,
          class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename std::enable_if<is_genfloat<_Up>::value,
                            complex<typename cplex::detail::__promote<_Tp, _Up>::type>>::type
    pow(const _Tp &__x, const complex<_Up> &__y) {
  typedef complex<typename cplex::detail::__promote<_Tp, _Up>::type> result_type;
  return pow(result_type(__x), result_type(__y));
}

namespace cplex::detail {
// __sqr, computes pow(x, 2)

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp> __sqr(const complex<_Tp> &__x) {
  return complex<_Tp>((__x.real() - __x.imag()) * (__x.real() + __x.imag()),
                      _Tp(2) * __x.real() * __x.imag());
}
}

// asinh

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp> asinh(const complex<_Tp> &__x) {
  const _Tp __pi(sycl::atan2(+0., -0.));
  if (sycl::isinf(__x.real())) {
    if (sycl::isnan(__x.imag()))
      return __x;
    if (sycl::isinf(__x.imag()))
      return complex<_Tp>(__x.real(),
                          sycl::copysign(__pi * _Tp(0.25), __x.imag()));
    return complex<_Tp>(__x.real(), sycl::copysign(_Tp(0), __x.imag()));
  }
  if (sycl::isnan(__x.real())) {
    if (sycl::isinf(__x.imag()))
      return complex<_Tp>(__x.imag(), __x.real());
    if (__x.imag() == 0)
      return __x;
    return complex<_Tp>(__x.real(), __x.real());
  }
  if (sycl::isinf(__x.imag()))
    return complex<_Tp>(sycl::copysign(__x.imag(), __x.real()),
                        sycl::copysign(__pi / _Tp(2), __x.imag()));
  complex<_Tp> __z = log(__x + sqrt(cplex::detail::__sqr(__x) + _Tp(1)));
  return complex<_Tp>(sycl::copysign(__z.real(), __x.real()),
                      sycl::copysign(__z.imag(), __x.imag()));
}

// acosh

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp> acosh(const complex<_Tp> &__x) {
  const _Tp __pi(sycl::atan2(+0., -0.));
  if (sycl::isinf(__x.real())) {
    if (sycl::isnan(__x.imag()))
      return complex<_Tp>(sycl::fabs(__x.real()), __x.imag());
    if (sycl::isinf(__x.imag())) {
      if (__x.real() > 0)
        return complex<_Tp>(__x.real(),
                            sycl::copysign(__pi * _Tp(0.25), __x.imag()));
      else
        return complex<_Tp>(-__x.real(),
                            sycl::copysign(__pi * _Tp(0.75), __x.imag()));
    }
    if (__x.real() < 0)
      return complex<_Tp>(-__x.real(), sycl::copysign(__pi, __x.imag()));
    return complex<_Tp>(__x.real(), sycl::copysign(_Tp(0), __x.imag()));
  }
  if (sycl::isnan(__x.real())) {
    if (sycl::isinf(__x.imag()))
      return complex<_Tp>(sycl::fabs(__x.imag()), __x.real());
    return complex<_Tp>(__x.real(), __x.real());
  }
  if (sycl::isinf(__x.imag()))
    return complex<_Tp>(sycl::fabs(__x.imag()),
                        sycl::copysign(__pi / _Tp(2), __x.imag()));
  complex<_Tp> __z = log(__x + sqrt(cplex::detail::__sqr(__x) - _Tp(1)));
  return complex<_Tp>(sycl::copysign(__z.real(), _Tp(0)),
                      sycl::copysign(__z.imag(), __x.imag()));
}

// atanh

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp> atanh(const complex<_Tp> &__x) {
  const _Tp __pi(sycl::atan2(+0., -0.));
  if (sycl::isinf(__x.imag())) {
    return complex<_Tp>(sycl::copysign(_Tp(0), __x.real()),
                        sycl::copysign(__pi / _Tp(2), __x.imag()));
  }
  if (sycl::isnan(__x.imag())) {
    if (sycl::isinf(__x.real()) || __x.real() == 0)
      return complex<_Tp>(sycl::copysign(_Tp(0), __x.real()), __x.imag());
    return complex<_Tp>(__x.imag(), __x.imag());
  }
  if (sycl::isnan(__x.real())) {
    return complex<_Tp>(__x.real(), __x.real());
  }
  if (sycl::isinf(__x.real())) {
    return complex<_Tp>(sycl::copysign(_Tp(0), __x.real()),
                        sycl::copysign(__pi / _Tp(2), __x.imag()));
  }
  if (sycl::fabs(__x.real()) == _Tp(1) && __x.imag() == _Tp(0)) {
    return complex<_Tp>(sycl::copysign(_Tp(INFINITY), __x.real()),
                        sycl::copysign(_Tp(0), __x.imag()));
  }
  complex<_Tp> __z = log((_Tp(1) + __x) / (_Tp(1) - __x)) / _Tp(2);
  return complex<_Tp>(sycl::copysign(__z.real(), __x.real()),
                      sycl::copysign(__z.imag(), __x.imag()));
}

// sinh

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp> sinh(const complex<_Tp> &__x) {
  if (sycl::isinf(__x.real()) && !sycl::isfinite(__x.imag()))
    return complex<_Tp>(__x.real(), _Tp(NAN));
  if (__x.real() == 0 && !sycl::isfinite(__x.imag()))
    return complex<_Tp>(__x.real(), _Tp(NAN));
  if (__x.imag() == 0 && !sycl::isfinite(__x.real()))
    return __x;
  return complex<_Tp>(sycl::sinh(__x.real()) * sycl::cos(__x.imag()),
                      sycl::cosh(__x.real()) * sycl::sin(__x.imag()));
}

// cosh

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp> cosh(const complex<_Tp> &__x) {
  if (sycl::isinf(__x.real()) && !sycl::isfinite(__x.imag()))
    return complex<_Tp>(sycl::fabs(__x.real()), _Tp(NAN));
  if (__x.real() == 0 && !sycl::isfinite(__x.imag()))
    return complex<_Tp>(_Tp(NAN), __x.real());
  if (__x.real() == 0 && __x.imag() == 0)
    return complex<_Tp>(_Tp(1), __x.imag());
  if (__x.imag() == 0 && !sycl::isfinite(__x.real()))
    return complex<_Tp>(sycl::fabs(__x.real()), __x.imag());
  return complex<_Tp>(sycl::cosh(__x.real()) * sycl::cos(__x.imag()),
                      sycl::sinh(__x.real()) * sycl::sin(__x.imag()));
}

// tanh

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp> tanh(const complex<_Tp> &__x) {
  if (sycl::isinf(__x.real())) {
    if (!sycl::isfinite(__x.imag()))
      return complex<_Tp>(sycl::copysign(_Tp(1), __x.real()), _Tp(0));
    return complex<_Tp>(sycl::copysign(_Tp(1), __x.real()),
                        sycl::copysign(_Tp(0), sycl::sin(_Tp(2) * __x.imag())));
  }
  if (sycl::isnan(__x.real()) && __x.imag() == 0)
    return __x;
  _Tp __2r(_Tp(2) * __x.real());
  _Tp __2i(_Tp(2) * __x.imag());
  _Tp __d(sycl::cosh(__2r) + sycl::cos(__2i));
  _Tp __2rsh(sycl::sinh(__2r));
  if (sycl::isinf(__2rsh) && sycl::isinf(__d))
    return complex<_Tp>(__2rsh > _Tp(0) ? _Tp(1) : _Tp(-1),
                        __2i > _Tp(0) ? _Tp(0) : _Tp(-0.));
  return complex<_Tp>(__2rsh / __d, sycl::sin(__2i) / __d);
}

// asin

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp> asin(const complex<_Tp> &__x) {
  complex<_Tp> __z = asinh(complex<_Tp>(-__x.imag(), __x.real()));
  return complex<_Tp>(__z.imag(), -__z.real());
}

// acos

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp> acos(const complex<_Tp> &__x) {
  const _Tp __pi(sycl::atan2(+0., -0.));
  if (sycl::isinf(__x.real())) {
    if (sycl::isnan(__x.imag()))
      return complex<_Tp>(__x.imag(), __x.real());
    if (sycl::isinf(__x.imag())) {
      if (__x.real() < _Tp(0))
        return complex<_Tp>(_Tp(0.75) * __pi, -__x.imag());
      return complex<_Tp>(_Tp(0.25) * __pi, -__x.imag());
    }
    if (__x.real() < _Tp(0))
      return complex<_Tp>(__pi,
                          sycl::signbit(__x.imag()) ? -__x.real() : __x.real());
    return complex<_Tp>(_Tp(0),
                        sycl::signbit(__x.imag()) ? __x.real() : -__x.real());
  }
  if (sycl::isnan(__x.real())) {
    if (sycl::isinf(__x.imag()))
      return complex<_Tp>(__x.real(), -__x.imag());
    return complex<_Tp>(__x.real(), __x.real());
  }
  if (sycl::isinf(__x.imag()))
    return complex<_Tp>(__pi / _Tp(2), -__x.imag());
  if (__x.real() == 0 && (__x.imag() == 0 || isnan(__x.imag())))
    return complex<_Tp>(__pi / _Tp(2), -__x.imag());
  complex<_Tp> __z = log(__x + sqrt(cplex::detail::__sqr(__x) - _Tp(1)));
  if (sycl::signbit(__x.imag()))
    return complex<_Tp>(sycl::fabs(__z.imag()), sycl::fabs(__z.real()));
  return complex<_Tp>(sycl::fabs(__z.imag()), -sycl::fabs(__z.real()));
}

// atan

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp> atan(const complex<_Tp> &__x) {
  complex<_Tp> __z = atanh(complex<_Tp>(-__x.imag(), __x.real()));
  return complex<_Tp>(__z.imag(), -__z.real());
}

// sin

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp> sin(const complex<_Tp> &__x) {
  complex<_Tp> __z = sinh(complex<_Tp>(-__x.imag(), __x.real()));
  return complex<_Tp>(__z.imag(), -__z.real());
}

// cos

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp> cos(const complex<_Tp> &__x) {
  return cosh(complex<_Tp>(-__x.imag(), __x.real()));
}

// tan

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp> tan(const complex<_Tp> &__x) {
  complex<_Tp> __z = tanh(complex<_Tp>(-__x.imag(), __x.real()));
  return complex<_Tp>(__z.imag(), -__z.real());
}

_SYCL_EXT_CPLX_END_NAMESPACE_STD

#undef _SYCL_EXT_CPLX_BEGIN_NAMESPACE_STD
#undef _SYCL_EXT_CPLX_END_NAMESPACE_STD
#undef _SYCL_EXT_CPLX_INLINE_VISIBILITY

#endif // _SYCL_EXT_CPLX_COMPLEX
