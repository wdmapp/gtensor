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

#ifndef _SYCL_MARRAY_NAMESPACE
#ifdef __HIPSYCL__
#define _SYCL_MARRAY_NAMESPACE hipsycl::sycl
#else
#define _SYCL_MARRAY_NAMESPACE sycl
#endif
#endif

#define _SYCL_MARRAY_BEGIN_NAMESPACE namespace _SYCL_MARRAY_NAMESPACE {
#define _SYCL_MARRAY_END_NAMESPACE }

#if defined(__FAST_MATH__) || defined(_M_FP_FAST)
#define _SYCL_EXT_CPLX_FAST_MATH
#endif

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

// Define our own fast-math aware wrappers for these routines, because
// some compilers are not able to perform the appropriate optimization
// without this extra help.
template <typename T>
_SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr bool isnan(const T a) {
#ifdef _SYCL_EXT_CPLX_FAST_MATH
  return false;
#else
  return sycl::isnan(a);
#endif
}

template <typename T>
_SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr bool isfinite(const T a) {
#ifdef _SYCL_EXT_CPLX_FAST_MATH
  return true;
#else
  return sycl::isfinite(a);
#endif
}

template <typename T>
_SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr bool isinf(const T a) {
#ifdef _SYCL_EXT_CPLX_FAST_MATH
  return false;
#else
  return sycl::isinf(a);
#endif
}

// To ensure loop unrolling is done when processing dimensions.
template <size_t... Inds, class F>
void loop_impl(std::integer_sequence<size_t, Inds...>, F &&f) {
  (f(std::integral_constant<size_t, Inds>{}), ...);
}

template <size_t count, class F> void loop(F &&f) {
  loop_impl(std::make_index_sequence<count>{}, std::forward<F>(f));
}

} // namespace cplex::detail

////////////////////////////////////////////////////////////////////////////////
// COMPLEX IMPLEMENTATION
////////////////////////////////////////////////////////////////////////////////

template <class _Tp, class _Enable = void> class complex;

template <class _Tp>
struct is_gencomplex
    : std::integral_constant<bool,
                             std::is_same_v<_Tp, complex<double>> ||
                                 std::is_same_v<_Tp, complex<float>> ||
                                 std::is_same_v<_Tp, complex<sycl::half>>> {};
template <typename _Tp>
inline constexpr bool is_gencomplex_v = is_gencomplex<_Tp>::value;

template <class _Tp>
struct is_genfloat
    : std::integral_constant<bool, std::is_same_v<_Tp, double> ||
                                       std::is_same_v<_Tp, float> ||
                                       std::is_same_v<_Tp, sycl::half>> {};
template <typename _Tp>
inline constexpr bool is_genfloat_v = is_genfloat<_Tp>::value;

template <class _Tp>
class complex<_Tp, typename std::enable_if<is_genfloat<_Tp>::value>::type> {
public:
  typedef _Tp value_type;

private:
  value_type __re_;
  value_type __im_;

public:
  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr complex(
      value_type __re = value_type(), value_type __im = value_type())
      : __re_(__re), __im_(__im) {}

  template <typename _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr complex(const complex<_Xp> &__c)
      : __re_(__c.real()), __im_(__c.imag()) {}

  template <class _Xp, class = std::enable_if<is_genfloat<_Xp>::value>>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr complex(
      const std::complex<_Xp> &__c)
      : __re_(static_cast<value_type>(__c.real())),
        __im_(static_cast<value_type>(__c.imag())) {}

  template <class _Xp, class = std::enable_if<is_genfloat<_Xp>::value>>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr
  operator std::complex<_Xp>() const {
    return std::complex<_Xp>(static_cast<_Xp>(__re_), static_cast<_Xp>(__im_));
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr value_type real() const {
    return __re_;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr value_type imag() const {
    return __im_;
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY void real(value_type __re) { __re_ = __re; }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY void imag(value_type __im) { __im_ = __im; }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator=(value_type __re) {
    __re_ = __re;
    __im_ = value_type();
    return *this;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex &
  operator+=(complex<value_type> &__c, value_type __re) {
    __c.__re_ += __re;
    return __c;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex &
  operator-=(complex<value_type> &__c, value_type __re) {
    __c.__re_ -= __re;
    return __c;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex &
  operator*=(complex<value_type> &__c, value_type __re) {
    __c.__re_ *= __re;
    __c.__im_ *= __re;
    return __c;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex &
  operator/=(complex<value_type> &__c, value_type __re) {
    __c.__re_ /= __re;
    __c.__im_ /= __re;
    return __c;
  }

  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator=(const complex<_Xp> &__c) {
    __re_ = __c.real();
    __im_ = __c.imag();
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex &
  operator+=(complex<value_type> &__x, const complex<_Xp> &__y) {
    __x.__re_ += __y.real();
    __x.__im_ += __y.imag();
    return __x;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex &
  operator-=(complex<value_type> &__x, const complex<_Xp> &__y) {
    __x.__re_ -= __y.real();
    __x.__im_ -= __y.imag();
    return __x;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex &
  operator*=(complex<value_type> &__x, const complex<_Xp> &__y) {
    __x = __x * complex(__y.real(), __y.imag());
    return __x;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex &
  operator/=(complex<value_type> &__x, const complex<_Xp> &__y) {
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
    if (cplex::detail::isnan(__x) && cplex::detail::isnan(__y)) {
      bool __recalc = false;
      if (cplex::detail::isinf(__a) || cplex::detail::isinf(__b)) {
        __a = sycl::copysign(
            cplex::detail::isinf(__a) ? value_type(1) : value_type(0), __a);
        __b = sycl::copysign(
            cplex::detail::isinf(__b) ? value_type(1) : value_type(0), __b);
        if (cplex::detail::isnan(__c))
          __c = sycl::copysign(value_type(0), __c);
        if (cplex::detail::isnan(__d))
          __d = sycl::copysign(value_type(0), __d);
        __recalc = true;
      }
      if (cplex::detail::isinf(__c) || cplex::detail::isinf(__d)) {
        __c = sycl::copysign(
            cplex::detail::isinf(__c) ? value_type(1) : value_type(0), __c);
        __d = sycl::copysign(
            cplex::detail::isinf(__d) ? value_type(1) : value_type(0), __d);
        if (cplex::detail::isnan(__a))
          __a = sycl::copysign(value_type(0), __a);
        if (cplex::detail::isnan(__b))
          __b = sycl::copysign(value_type(0), __b);
        __recalc = true;
      }
      if (!__recalc &&
          (cplex::detail::isinf(__ac) || cplex::detail::isinf(__bd) ||
           cplex::detail::isinf(__ad) || cplex::detail::isinf(__bc))) {
        if (cplex::detail::isnan(__a))
          __a = sycl::copysign(value_type(0), __a);
        if (cplex::detail::isnan(__b))
          __b = sycl::copysign(value_type(0), __b);
        if (cplex::detail::isnan(__c))
          __c = sycl::copysign(value_type(0), __c);
        if (cplex::detail::isnan(__d))
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
#if defined(_SYCL_EXT_CPLX_FAST_MATH)
    // This implementation is around 20% faster for single precision, 5% for
    // double, at the expense of larger error in some cases, because no scaling
    // is done.
    value_type __a = __z.__re_;
    value_type __b = __z.__im_;
    value_type __c = __w.__re_;
    value_type __d = __w.__im_;
    value_type __r = __a * __c + __b * __d;
    value_type __n = __b * __b + __d * __d;
    value_type __x = __r / __n;
    value_type __y = (__b * __c - __a * __d) / __n;
    return complex<value_type>(__x, __y);
#else
    int __ilogbw = 0;
    value_type __a = __z.__re_;
    value_type __b = __z.__im_;
    value_type __c = __w.__re_;
    value_type __d = __w.__im_;
    value_type __logbw =
        sycl::logb(sycl::fmax(sycl::fabs(__c), sycl::fabs(__d)));
    if (cplex::detail::isfinite(__logbw)) {
      __ilogbw = static_cast<int>(__logbw);
      __c = sycl::ldexp(__c, -__ilogbw);
      __d = sycl::ldexp(__d, -__ilogbw);
    }
    value_type __denom = __c * __c + __d * __d;
    value_type __x = sycl::ldexp((__a * __c + __b * __d) / __denom, -__ilogbw);
    value_type __y = sycl::ldexp((__b * __c - __a * __d) / __denom, -__ilogbw);
    if (cplex::detail::isnan(__x) && cplex::detail::isnan(__y)) {
      if ((__denom == value_type(0)) &&
          (!cplex::detail::isnan(__a) || !cplex::detail::isnan(__b))) {
        __x = sycl::copysign(value_type(INFINITY), __c) * __a;
        __y = sycl::copysign(value_type(INFINITY), __c) * __b;
      } else if ((cplex::detail::isinf(__a) || cplex::detail::isinf(__b)) &&
                 cplex::detail::isfinite(__c) && cplex::detail::isfinite(__d)) {
        __a = sycl::copysign(
            cplex::detail::isinf(__a) ? value_type(1) : value_type(0), __a);
        __b = sycl::copysign(
            cplex::detail::isinf(__b) ? value_type(1) : value_type(0), __b);
        __x = value_type(INFINITY) * (__a * __c + __b * __d);
        __y = value_type(INFINITY) * (__b * __c - __a * __d);
      } else if (cplex::detail::isinf(__logbw) && __logbw > value_type(0) &&
                 cplex::detail::isfinite(__a) && cplex::detail::isfinite(__b)) {
        __c = sycl::copysign(
            cplex::detail::isinf(__c) ? value_type(1) : value_type(0), __c);
        __d = sycl::copysign(
            cplex::detail::isinf(__d) ? value_type(1) : value_type(0), __d);
        __x = value_type(0) * (__a * __c + __b * __d);
        __y = value_type(0) * (__b * __c - __a * __d);
      }
    }
    return complex<value_type>(__x, __y);
#endif
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
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend std::basic_istream<_CharT, _Traits> &
  operator>>(std::basic_istream<_CharT, _Traits> &__is,
             complex<value_type> &__x) {
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
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend std::basic_ostream<_CharT, _Traits> &
  operator<<(std::basic_ostream<_CharT, _Traits> &__os,
             const complex<value_type> &__x) {
    std::basic_ostringstream<_CharT, _Traits> __s;
    __s.flags(__os.flags());
    __s.imbue(__os.getloc());
    __s.precision(__os.precision());
    __s << '(' << __x.__re_ << ',' << __x.__im_ << ')';
    return __os << __s.str();
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend const sycl::stream &
  operator<<(const sycl::stream &__ss, const complex<value_type> &_x) {
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
} // namespace cplex::detail

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
    typename cplex::detail::__libcpp_complex_overload_traits<_Tp>::_ValueType
    arg(_Tp __re) {
  typedef
      typename cplex::detail::__libcpp_complex_overload_traits<_Tp>::_ValueType
          _ValueType;
  return sycl::atan2(static_cast<_ValueType>(0), static_cast<_ValueType>(__re));
}

// norm

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY _Tp norm(const complex<_Tp> &__c) {
  if (cplex::detail::isinf(__c.real()))
    return sycl::fabs(__c.real());
  if (cplex::detail::isinf(__c.imag()))
    return sycl::fabs(__c.imag());
  return __c.real() * __c.real() + __c.imag() * __c.imag();
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename cplex::detail::__libcpp_complex_overload_traits<_Tp>::_ValueType
    norm(_Tp __re) {
  typedef
      typename cplex::detail::__libcpp_complex_overload_traits<_Tp>::_ValueType
          _ValueType;
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
  typedef typename cplex::detail::__libcpp_complex_overload_traits<
      _Tp>::_ComplexType _ComplexType;
  return _ComplexType(__re);
}

// proj

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp> proj(const complex<_Tp> &__c) {
  complex<_Tp> __r = __c;
  if (cplex::detail::isinf(__c.real()) || cplex::detail::isinf(__c.imag()))
    __r = complex<_Tp>(INFINITY, sycl::copysign(_Tp(0), __c.imag()));
  return __r;
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename cplex::detail::__libcpp_complex_overload_traits<_Tp>::_ComplexType
    proj(_Tp __re) {
  typedef typename cplex::detail::__libcpp_complex_overload_traits<
      _Tp>::_ComplexType _ComplexType;

  if constexpr (!std::is_integral_v<_Tp>) {
    if (cplex::detail::isinf(__re))
      __re = sycl::fabs(__re);
  }

  return _ComplexType(__re);
}

// polar

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp>
polar(const _Tp &__rho, const _Tp &__theta = _Tp()) {
  if (cplex::detail::isnan(__rho) || sycl::signbit(__rho))
    return complex<_Tp>(_Tp(NAN), _Tp(NAN));
  if (cplex::detail::isnan(__theta)) {
    if (cplex::detail::isinf(__rho))
      return complex<_Tp>(__rho, __theta);
    return complex<_Tp>(__theta, __theta);
  }
  if (cplex::detail::isinf(__theta)) {
    if (cplex::detail::isinf(__rho))
      return complex<_Tp>(__rho, _Tp(NAN));
    return complex<_Tp>(_Tp(NAN), _Tp(NAN));
  }
  _Tp __x = __rho * sycl::cos(__theta);
  if (cplex::detail::isnan(__x))
    __x = 0;
  _Tp __y = __rho * sycl::sin(__theta);
  if (cplex::detail::isnan(__y))
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
  if (cplex::detail::isinf(__x.imag()))
    return complex<_Tp>(_Tp(INFINITY), __x.imag());
  if (cplex::detail::isinf(__x.real())) {
    if (__x.real() > _Tp(0))
      return complex<_Tp>(__x.real(), cplex::detail::isnan(__x.imag())
                                          ? __x.imag()
                                          : sycl::copysign(_Tp(0), __x.imag()));
    return complex<_Tp>(cplex::detail::isnan(__x.imag()) ? __x.imag() : _Tp(0),
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
  if (cplex::detail::isinf(__x.real())) {
    if (__x.real() < _Tp(0)) {
      if (!cplex::detail::isfinite(__i))
        __i = _Tp(1);
    } else if (__i == 0 || !cplex::detail::isfinite(__i)) {
      if (cplex::detail::isinf(__i))
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
_SYCL_EXT_CPLX_INLINE_VISIBILITY
    complex<typename cplex::detail::__promote<_Tp, _Up>::type>
    pow(const complex<_Tp> &__x, const complex<_Up> &__y) {
  typedef complex<typename cplex::detail::__promote<_Tp, _Up>::type>
      result_type;
  return pow(result_type(__x), result_type(__y));
}

template <class _Tp, class _Up,
          class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY typename std::enable_if<
    is_genfloat<_Up>::value,
    complex<typename cplex::detail::__promote<_Tp, _Up>::type>>::type
pow(const complex<_Tp> &__x, const _Up &__y) {
  typedef complex<typename cplex::detail::__promote<_Tp, _Up>::type>
      result_type;
  return pow(result_type(__x), result_type(__y));
}

template <class _Tp, class _Up,
          class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY typename std::enable_if<
    is_genfloat<_Up>::value,
    complex<typename cplex::detail::__promote<_Tp, _Up>::type>>::type
pow(const _Tp &__x, const complex<_Up> &__y) {
  typedef complex<typename cplex::detail::__promote<_Tp, _Up>::type>
      result_type;
  return pow(result_type(__x), result_type(__y));
}

namespace cplex::detail {
// __sqr, computes pow(x, 2)

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp> __sqr(const complex<_Tp> &__x) {
  return complex<_Tp>((__x.real() - __x.imag()) * (__x.real() + __x.imag()),
                      _Tp(2) * __x.real() * __x.imag());
}
} // namespace cplex::detail

// asinh

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp> asinh(const complex<_Tp> &__x) {
  const _Tp __pi(sycl::atan2(_Tp(+0.), _Tp(-0.)));
  if (cplex::detail::isinf(__x.real())) {
    if (cplex::detail::isnan(__x.imag()))
      return __x;
    if (cplex::detail::isinf(__x.imag()))
      return complex<_Tp>(__x.real(),
                          sycl::copysign(__pi * _Tp(0.25), __x.imag()));
    return complex<_Tp>(__x.real(), sycl::copysign(_Tp(0), __x.imag()));
  }
  if (cplex::detail::isnan(__x.real())) {
    if (cplex::detail::isinf(__x.imag()))
      return complex<_Tp>(__x.imag(), __x.real());
    if (__x.imag() == 0)
      return __x;
    return complex<_Tp>(__x.real(), __x.real());
  }
  if (cplex::detail::isinf(__x.imag()))
    return complex<_Tp>(sycl::copysign(__x.imag(), __x.real()),
                        sycl::copysign(__pi / _Tp(2), __x.imag()));
  complex<_Tp> __z = log(__x + sqrt(cplex::detail::__sqr(__x) + _Tp(1)));
  return complex<_Tp>(sycl::copysign(__z.real(), __x.real()),
                      sycl::copysign(__z.imag(), __x.imag()));
}

// acosh

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp> acosh(const complex<_Tp> &__x) {
  const _Tp __pi(sycl::atan2(_Tp(+0.), _Tp(-0.)));
  if (cplex::detail::isinf(__x.real())) {
    if (cplex::detail::isnan(__x.imag()))
      return complex<_Tp>(sycl::fabs(__x.real()), __x.imag());
    if (cplex::detail::isinf(__x.imag())) {
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
  if (cplex::detail::isnan(__x.real())) {
    if (cplex::detail::isinf(__x.imag()))
      return complex<_Tp>(sycl::fabs(__x.imag()), __x.real());
    return complex<_Tp>(__x.real(), __x.real());
  }
  if (cplex::detail::isinf(__x.imag()))
    return complex<_Tp>(sycl::fabs(__x.imag()),
                        sycl::copysign(__pi / _Tp(2), __x.imag()));
  complex<_Tp> __z = log(__x + sqrt(cplex::detail::__sqr(__x) - _Tp(1)));
  return complex<_Tp>(sycl::copysign(__z.real(), _Tp(0)),
                      sycl::copysign(__z.imag(), __x.imag()));
}

// atanh

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp> atanh(const complex<_Tp> &__x) {
  const _Tp __pi(sycl::atan2(_Tp(+0.), _Tp(-0.)));
  if (cplex::detail::isinf(__x.imag())) {
    return complex<_Tp>(sycl::copysign(_Tp(0), __x.real()),
                        sycl::copysign(__pi / _Tp(2), __x.imag()));
  }
  if (cplex::detail::isnan(__x.imag())) {
    if (cplex::detail::isinf(__x.real()) || __x.real() == 0)
      return complex<_Tp>(sycl::copysign(_Tp(0), __x.real()), __x.imag());
    return complex<_Tp>(__x.imag(), __x.imag());
  }
  if (cplex::detail::isnan(__x.real())) {
    return complex<_Tp>(__x.real(), __x.real());
  }
  if (cplex::detail::isinf(__x.real())) {
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
  if (cplex::detail::isinf(__x.real()) && !cplex::detail::isfinite(__x.imag()))
    return complex<_Tp>(__x.real(), _Tp(NAN));
  if (__x.real() == 0 && !cplex::detail::isfinite(__x.imag()))
    return complex<_Tp>(__x.real(), _Tp(NAN));
  if (__x.imag() == 0 && !cplex::detail::isfinite(__x.real()))
    return __x;
  return complex<_Tp>(sycl::sinh(__x.real()) * sycl::cos(__x.imag()),
                      sycl::cosh(__x.real()) * sycl::sin(__x.imag()));
}

// cosh

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp> cosh(const complex<_Tp> &__x) {
  if (cplex::detail::isinf(__x.real()) && !cplex::detail::isfinite(__x.imag()))
    return complex<_Tp>(sycl::fabs(__x.real()), _Tp(NAN));
  if (__x.real() == 0 && !cplex::detail::isfinite(__x.imag()))
    return complex<_Tp>(_Tp(NAN), __x.real());
  if (__x.real() == 0 && __x.imag() == 0)
    return complex<_Tp>(_Tp(1), __x.imag());
  if (__x.imag() == 0 && !cplex::detail::isfinite(__x.real()))
    return complex<_Tp>(sycl::fabs(__x.real()), __x.imag());
  return complex<_Tp>(sycl::cosh(__x.real()) * sycl::cos(__x.imag()),
                      sycl::sinh(__x.real()) * sycl::sin(__x.imag()));
}

// tanh

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp> tanh(const complex<_Tp> &__x) {
  if (cplex::detail::isinf(__x.real())) {
    if (!cplex::detail::isfinite(__x.imag()))
      return complex<_Tp>(sycl::copysign(_Tp(1), __x.real()), _Tp(0));
    return complex<_Tp>(sycl::copysign(_Tp(1), __x.real()),
                        sycl::copysign(_Tp(0), sycl::sin(_Tp(2) * __x.imag())));
  }
  if (cplex::detail::isnan(__x.real()) && __x.imag() == 0)
    return __x;
  _Tp __2r(_Tp(2) * __x.real());
  _Tp __2i(_Tp(2) * __x.imag());
  _Tp __d(sycl::cosh(__2r) + sycl::cos(__2i));
  _Tp __2rsh(sycl::sinh(__2r));
  if (cplex::detail::isinf(__2rsh) && cplex::detail::isinf(__d))
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
  const _Tp __pi(sycl::atan2(_Tp(+0.), _Tp(-0.)));
  if (cplex::detail::isinf(__x.real())) {
    if (cplex::detail::isnan(__x.imag()))
      return complex<_Tp>(__x.imag(), __x.real());
    if (cplex::detail::isinf(__x.imag())) {
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
  if (cplex::detail::isnan(__x.real())) {
    if (cplex::detail::isinf(__x.imag()))
      return complex<_Tp>(__x.real(), -__x.imag());
    return complex<_Tp>(__x.real(), __x.real());
  }
  if (cplex::detail::isinf(__x.imag()))
    return complex<_Tp>(__pi / _Tp(2), -__x.imag());
  if (__x.real() == 0 && (__x.imag() == 0 || cplex::detail::isnan(__x.imag())))
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

////////////////////////////////////////////////////////////////////////////////
// MARRAY IMPLEMENTATION
////////////////////////////////////////////////////////////////////////////////

_SYCL_EXT_CPLX_BEGIN_NAMESPACE_STD

template <typename T> struct is_mgencomplex : std::false_type {};

template <typename T, std::size_t N>
struct is_mgencomplex<sycl::marray<T, N>>
    : std::integral_constant<bool, sycl::ext::cplx::is_gencomplex_v<T>> {};

template <typename T>
inline constexpr bool is_mgencomplex_v = is_mgencomplex<T>::value;

_SYCL_EXT_CPLX_END_NAMESPACE_STD

_SYCL_MARRAY_BEGIN_NAMESPACE

// marray of complex class specialisation
template <typename T, std::size_t NumElements>
class marray<sycl::ext::cplx::complex<T>, NumElements> {
private:
  using ComplexDataT = sycl::ext::cplx::complex<T>;

public:
  using value_type = ComplexDataT;
  using reference = ComplexDataT &;
  using const_reference = const ComplexDataT &;
  using iterator = ComplexDataT *;
  using const_iterator = const ComplexDataT *;

private:
  value_type MData[NumElements];

public:
  constexpr marray() : MData{} {};

  explicit constexpr marray(const ComplexDataT &arg) {
    for (size_t i = 0; i < NumElements; ++i)
      MData[i] = arg;
  }

  template <typename... ArgTN>
  constexpr marray(const ArgTN &...args) : MData{args...} {};

  constexpr marray(const marray<ComplexDataT, NumElements> &rhs) = default;
  constexpr marray(marray<ComplexDataT, NumElements> &&rhs) = default;

  // Available only when: NumElements == 1
  template <typename = typename std::enable_if<NumElements == 1>>
  operator ComplexDataT() const {
    return MData[0];
  }

  static constexpr std::size_t size() noexcept { return NumElements; }

  marray<T, NumElements> real() const {
    sycl::marray<T, NumElements> rtn;

    for (std::size_t i = 0; i < NumElements; ++i) {
      rtn[i] = MData[i].real();
    }

    return rtn;
  }

  marray<T, NumElements> imag() const {
    sycl::marray<T, NumElements> rtn;

    for (std::size_t i = 0; i < NumElements; ++i) {
      rtn[i] = MData[i].imag();
    }

    return rtn;
  }

  // subscript operator
  reference operator[](std::size_t i) { return MData[i]; }
  const_reference operator[](std::size_t i) const { return MData[i]; }

  marray &operator=(const marray<ComplexDataT, NumElements> &rhs) = default;
  marray &operator=(const ComplexDataT &rhs) {
    for (std::size_t i = 0; i < NumElements; ++i)
      MData[i] = rhs;

    return *this;
  }

  // iterator functions
  iterator begin() { return MData; }
  const_iterator begin() const { return MData; }

  iterator end() { return MData + NumElements; }
  const_iterator end() const { return MData + NumElements; }

  // OP is: +, -, *, /
#define OP(op)                                                                 \
  friend marray operator op(const marray &lhs, const marray &rhs) {            \
    marray rtn;                                                                \
    for (std::size_t i = 0; i < NumElements; ++i)                              \
      rtn[i] = lhs[i] op rhs[i];                                               \
                                                                               \
    return rtn;                                                                \
  }                                                                            \
                                                                               \
  friend marray operator op(const marray &lhs, const ComplexDataT &rhs) {      \
    marray rtn;                                                                \
    for (std::size_t i = 0; i < NumElements; ++i)                              \
      rtn[i] = lhs[i] op rhs;                                                  \
                                                                               \
    return rtn;                                                                \
  }                                                                            \
                                                                               \
  friend marray operator op(const ComplexDataT &lhs, const marray &rhs) {      \
    marray rtn;                                                                \
    for (std::size_t i = 0; i < NumElements; ++i)                              \
      rtn[i] = lhs op rhs[i];                                                  \
                                                                               \
    return rtn;                                                                \
  }

  OP(+)
  OP(-)
  OP(*)
  OP(/)

#undef OP

  // OP is: %
  friend marray operator%(const marray &lhs, const marray &rhs) = delete;
  friend marray operator%(const marray &lhs, const ComplexDataT &rhs) = delete;
  friend marray operator%(const ComplexDataT &lhs, const marray &rhs) = delete;

  // OP is: +=, -=, *=, /=
#define OP(op)                                                                 \
  friend marray &operator op(marray &lhs, const marray &rhs) {                 \
    for (std::size_t i = 0; i < NumElements; ++i)                              \
      lhs[i] op rhs[i];                                                        \
                                                                               \
    return lhs;                                                                \
  }                                                                            \
                                                                               \
  friend marray &operator op(marray &lhs, const ComplexDataT &rhs) {           \
    for (std::size_t i = 0; i < NumElements; ++i)                              \
      lhs[i] op rhs;                                                           \
                                                                               \
    return lhs;                                                                \
  }                                                                            \
  friend marray &operator op(ComplexDataT &lhs, const marray &rhs) {           \
    for (std::size_t i = 0; i < NumElements; ++i)                              \
      lhs[i] op rhs;                                                           \
                                                                               \
    return lhs;                                                                \
  }

  OP(+=)
  OP(-=)
  OP(*=)
  OP(/=)

#undef OP

  // OP is: %=
  friend marray &operator%=(marray &lhs, const marray &rhs) = delete;
  friend marray &operator%=(marray &lhs, const ComplexDataT &rhs) = delete;
  friend marray &operator%=(ComplexDataT &lhs, const marray &rhs) = delete;

// OP is: ++, --
#define OP(op)                                                                 \
  friend marray operator op(marray &lhs, int) = delete;                        \
  friend marray &operator op(marray &rhs) = delete;

  OP(++)
  OP(--)

#undef OP

// OP is: unary +, unary -
#define OP(op)                                                                 \
  friend marray<ComplexDataT, NumElements> operator op(                        \
      const marray<ComplexDataT, NumElements> &rhs) {                          \
    marray<ComplexDataT, NumElements> rtn;                                     \
                                                                               \
    for (std::size_t i = 0; i < NumElements; ++i) {                            \
      rtn[i] = op rhs[i];                                                      \
    }                                                                          \
                                                                               \
    return rtn;                                                                \
  }

  OP(+)
  OP(-)

#undef OP

// OP is: &, |, ^
#define OP(op)                                                                 \
  friend marray operator op(const marray &lhs, const marray &rhs) = delete;    \
  friend marray operator op(const marray &lhs, const ComplexDataT &rhs) =      \
      delete;

  OP(&)
  OP(|)
  OP(^)

#undef OP

// OP is: &=, |=, ^=
#define OP(op)                                                                 \
  friend marray &operator op(marray &lhs, const marray &rhs) = delete;         \
  friend marray &operator op(marray &lhs, const ComplexDataT &rhs) = delete;   \
  friend marray &operator op(ComplexDataT &lhs, const marray &rhs) = delete;

  OP(&=)
  OP(|=)
  OP(^=)

#undef OP

// OP is: &&, ||
#define OP(op)                                                                 \
  friend marray<bool, NumElements> operator op(const marray &lhs,              \
                                               const marray &rhs) = delete;    \
  friend marray<bool, NumElements> operator op(                                \
      const marray &lhs, const ComplexDataT &rhs) = delete;                    \
  friend marray<bool, NumElements> operator op(const ComplexDataT &lhs,        \
                                               const marray &rhs) = delete;

  OP(&&)
  OP(||)

#undef OP

// OP is: <<, >>
#define OP(op)                                                                 \
  friend marray operator op(const marray &lhs, const marray &rhs) = delete;    \
  friend marray operator op(const marray &lhs, const ComplexDataT &rhs) =      \
      delete;                                                                  \
  friend marray operator op(const ComplexDataT &lhs, const marray &rhs) =      \
      delete;

  OP(<<)
  OP(>>)

#undef OP

// OP is: <<=, >>=
#define OP(op)                                                                 \
  friend marray &operator op(marray &lhs, const marray &rhs) = delete;         \
  friend marray &operator op(marray &lhs, const ComplexDataT &rhs) = delete;

  OP(<<=)
  OP(>>=)

#undef OP

  // OP is: ==, !=
#define OP(op)                                                                 \
  friend marray<bool, NumElements> operator op(const marray &lhs,              \
                                               const marray &rhs) {            \
    marray<bool, NumElements> rtn;                                             \
    for (std::size_t i = 0; i < NumElements; ++i)                              \
      rtn[i] = lhs[i] op rhs[i];                                               \
                                                                               \
    return rtn;                                                                \
  }                                                                            \
                                                                               \
  friend marray<bool, NumElements> operator op(const marray &lhs,              \
                                               const ComplexDataT &rhs) {      \
    marray<bool, NumElements> rtn;                                             \
    for (std::size_t i = 0; i < NumElements; ++i)                              \
      rtn[i] = lhs[i] op rhs;                                                  \
                                                                               \
    return rtn;                                                                \
  }                                                                            \
                                                                               \
  friend marray<bool, NumElements> operator op(const ComplexDataT &lhs,        \
                                               const marray &rhs) {            \
    marray<bool, NumElements> rtn;                                             \
    for (std::size_t i = 0; i < NumElements; ++i)                              \
      rtn[i] = lhs op rhs[i];                                                  \
                                                                               \
    return rtn;                                                                \
  }

  OP(==)
  OP(!=)

#undef OP

  // OP is: <, >, <=, >=
#define OP(op)                                                                 \
  friend marray<bool, NumElements> operator op(const marray &lhs,              \
                                               const marray &rhs) = delete;    \
  friend marray<bool, NumElements> operator op(                                \
      const marray &lhs, const ComplexDataT &rhs) = delete;                    \
  friend marray<bool, NumElements> operator op(const ComplexDataT &lhs,        \
                                               const marray &rhs) = delete;

  OP(<);
  OP(>);
  OP(<=);
  OP(>=);

#undef OP

  friend marray operator~(const marray &v) = delete;

  friend marray<bool, NumElements> operator!(const marray &v) = delete;
};

_SYCL_MARRAY_END_NAMESPACE

_SYCL_EXT_CPLX_BEGIN_NAMESPACE_STD

// Math marray overloads

#define MATH_OP_ONE_PARAM(math_func, rtn_type, arg_type)                       \
  template <typename T, std::size_t NumElements,                               \
            typename = std::enable_if<is_genfloat<T>::value ||                 \
                                      is_gencomplex<T>::value>>                \
  _SYCL_EXT_CPLX_INLINE_VISIBILITY sycl::marray<rtn_type, NumElements>         \
  math_func(const sycl::marray<arg_type, NumElements> &x) {                    \
    sycl::marray<rtn_type, NumElements> rtn;                                   \
    for (std::size_t i = 0; i < NumElements; ++i)                              \
      rtn[i] = sycl::ext::cplx::math_func(x[i]);                               \
                                                                               \
    return rtn;                                                                \
  }

MATH_OP_ONE_PARAM(abs, T, complex<T>);
MATH_OP_ONE_PARAM(acos, complex<T>, complex<T>);
MATH_OP_ONE_PARAM(asin, complex<T>, complex<T>);
MATH_OP_ONE_PARAM(atan, complex<T>, complex<T>);
MATH_OP_ONE_PARAM(acosh, complex<T>, complex<T>);
MATH_OP_ONE_PARAM(asinh, complex<T>, complex<T>);
MATH_OP_ONE_PARAM(atanh, complex<T>, complex<T>);
MATH_OP_ONE_PARAM(arg, T, complex<T>);
MATH_OP_ONE_PARAM(conj, complex<T>, complex<T>);
MATH_OP_ONE_PARAM(cos, complex<T>, complex<T>);
MATH_OP_ONE_PARAM(cosh, complex<T>, complex<T>);
MATH_OP_ONE_PARAM(exp, complex<T>, complex<T>);
MATH_OP_ONE_PARAM(log, complex<T>, complex<T>);
MATH_OP_ONE_PARAM(log10, complex<T>, complex<T>);
MATH_OP_ONE_PARAM(norm, T, complex<T>);
MATH_OP_ONE_PARAM(proj, complex<T>, complex<T>);
MATH_OP_ONE_PARAM(proj, complex<T>, T);
MATH_OP_ONE_PARAM(sin, complex<T>, complex<T>);
MATH_OP_ONE_PARAM(sinh, complex<T>, complex<T>);
MATH_OP_ONE_PARAM(sqrt, complex<T>, complex<T>);
MATH_OP_ONE_PARAM(tan, complex<T>, complex<T>);
MATH_OP_ONE_PARAM(tanh, complex<T>, complex<T>);

#undef MATH_OP_ONE_PARAM

#define MATH_OP_TWO_PARAM(math_func, rtn_type, arg_type1, arg_type2)           \
  template <typename T, std::size_t NumElements,                               \
            typename = std::enable_if<is_genfloat<T>::value ||                 \
                                      is_gencomplex<T>::value>>                \
  _SYCL_EXT_CPLX_INLINE_VISIBILITY sycl::marray<rtn_type, NumElements>         \
  math_func(const sycl::marray<arg_type1, NumElements> &x,                     \
            const sycl::marray<arg_type2, NumElements> &y) {                   \
    sycl::marray<rtn_type, NumElements> rtn;                                   \
    for (std::size_t i = 0; i < NumElements; ++i)                              \
      rtn[i] = sycl::ext::cplx::math_func(x[i], y[i]);                         \
                                                                               \
    return rtn;                                                                \
  }                                                                            \
                                                                               \
  template <typename T, std::size_t NumElements,                               \
            typename = std::enable_if<is_genfloat<T>::value ||                 \
                                      is_gencomplex<T>::value>>                \
  _SYCL_EXT_CPLX_INLINE_VISIBILITY sycl::marray<rtn_type, NumElements>         \
  math_func(const sycl::marray<arg_type1, NumElements> &x,                     \
            const arg_type2 &y) {                                              \
    sycl::marray<rtn_type, NumElements> rtn;                                   \
    for (std::size_t i = 0; i < NumElements; ++i)                              \
      rtn[i] = sycl::ext::cplx::math_func(x[i], y);                            \
                                                                               \
    return rtn;                                                                \
  }                                                                            \
                                                                               \
  template <typename T, std::size_t NumElements,                               \
            typename = std::enable_if<is_genfloat<T>::value ||                 \
                                      is_gencomplex<T>::value>>                \
  _SYCL_EXT_CPLX_INLINE_VISIBILITY sycl::marray<rtn_type, NumElements>         \
  math_func(const arg_type1 &x,                                                \
            const sycl::marray<arg_type2, NumElements> &y) {                   \
    sycl::marray<rtn_type, NumElements> rtn;                                   \
    for (std::size_t i = 0; i < NumElements; ++i)                              \
      rtn[i] = math_func(x, y[i]);                                             \
                                                                               \
    return rtn;                                                                \
  }

MATH_OP_TWO_PARAM(pow, complex<T>, complex<T>, T);
MATH_OP_TWO_PARAM(pow, complex<T>, complex<T>, complex<T>);
MATH_OP_TWO_PARAM(pow, complex<T>, T, complex<T>);

#undef MATH_OP_TWO_PARAM

// Special definition as polar requires default argument

template <typename T, std::size_t NumElements,
          typename = std::enable_if<is_genfloat<T>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY
    sycl::marray<sycl::ext::cplx::complex<T>, NumElements>
    polar(const sycl::marray<T, NumElements> &rho,
          const sycl::marray<T, NumElements> &theta) {
  sycl::marray<sycl::ext::cplx::complex<T>, NumElements> rtn;
  for (std::size_t i = 0; i < NumElements; ++i)
    rtn[i] = sycl::ext::cplx::polar(rho[i], theta[i]);

  return rtn;
}

template <typename T, std::size_t NumElements,
          typename = std::enable_if<is_genfloat<T>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY
    sycl::marray<sycl::ext::cplx::complex<T>, NumElements>
    polar(const sycl::marray<T, NumElements> &rho, const T &theta = 0) {
  sycl::marray<sycl::ext::cplx::complex<T>, NumElements> rtn;
  for (std::size_t i = 0; i < NumElements; ++i)
    rtn[i] = sycl::ext::cplx::polar(rho[i], theta);

  return rtn;
}

template <typename T, std::size_t NumElements,
          typename = std::enable_if<is_genfloat<T>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY
    sycl::marray<sycl::ext::cplx::complex<T>, NumElements>
    polar(const T &rho, const sycl::marray<T, NumElements> &theta) {
  sycl::marray<sycl::ext::cplx::complex<T>, NumElements> rtn;
  for (std::size_t i = 0; i < NumElements; ++i)
    rtn[i] = sycl::ext::cplx::polar(rho, theta[i]);

  return rtn;
}

////////////////////////////////////////////////////////////////////////////////
// GROUP ALGORITMHS
////////////////////////////////////////////////////////////////////////////////

namespace cplex::detail {

/// Helper traits to check if the type is a sycl::plus
template <typename BinaryOperation>
struct is_plus
    : std::integral_constant<bool,
                             std::is_same_v<BinaryOperation, std::plus<void>>> {
};
template <typename BinaryOperation>
inline constexpr bool is_plus_v = is_plus<BinaryOperation>::value;

/// Helper traits to check if the type is a sycl:multiplies
template <typename BinaryOperation>
struct is_multiplies
    : std::integral_constant<
          bool, std::is_same_v<BinaryOperation, std::multiplies<void>>> {};
template <typename BinaryOperation>
inline constexpr bool is_multiplies_v = is_multiplies<BinaryOperation>::value;

/// Wrapper trait to check if the binary operation is supported
template <typename BinaryOperation>
struct is_binary_op_supported
    : std::integral_constant<bool,
                             (detail::is_plus<BinaryOperation>::value ||
                              detail::is_multiplies<BinaryOperation>::value)> {
};
template <class BinaryOperation>
inline constexpr bool is_binary_op_supported_v =
    is_binary_op_supported<BinaryOperation>::value;

/// Helper functions to get the init for sycl::plus binary operation when the
/// type is a gencomplex
template <typename T, class BinaryOperation>
std::enable_if_t<(sycl::ext::cplx::is_gencomplex_v<T> &&
                  detail::is_plus_v<BinaryOperation>),
                 T>
get_init() {
  return T{0, 0};
}
/// Helper functions to get the init for sycl::multiply binary operation when
/// the type is a gencomplex
template <typename T, class BinaryOperation>
std::enable_if_t<(sycl::ext::cplx::is_gencomplex_v<T> &&
                  detail::is_multiplies<BinaryOperation>::value),
                 T>
get_init() {
  return T{1, 0};
}
/// Helper functions to get the init for sycl::plus binary operation when the
/// type is a mgencomplex
template <typename T, class BinaryOperation>
std::enable_if_t<
    (is_mgencomplex_v<T> && detail::is_plus<BinaryOperation>::value), T>
get_init() {
  using Complex = typename T::value_type;

  T result;
  std::fill(result.begin(), result.end(), Complex{0, 0});
  return result;
}
/// Helper functions to get the init for sycl::multiply binary operation when
/// the type is a mgencomplex
template <typename T, class BinaryOperation>
std::enable_if_t<
    (is_mgencomplex_v<T> && detail::is_multiplies<BinaryOperation>::value), T>
get_init() {
  using Complex = typename T::value_type;

  T result;
  std::fill(result.begin(), result.end(), Complex{1, 0});
  return result;
}

} // namespace cplex::detail

/* REDUCE_OVER_GROUP'S OVERLOADS */

/// Complex specialization
template <typename Group, typename V, typename T, class BinaryOperation,
          typename = std::enable_if_t<
              sycl::is_group_v<std::decay_t<Group>> && is_genfloat_v<V> &&
              is_genfloat_v<T> &&
              cplex::detail::is_binary_op_supported_v<BinaryOperation>>>
complex<T> reduce_over_group(Group g, complex<V> x, complex<T> init,
                             BinaryOperation binary_op) {
#ifdef __SYCL_DEVICE_ONLY__
  complex<T> result;

  result.real(sycl::reduce_over_group(g, x.real(), init.real(), binary_op));
  result.imag(sycl::reduce_over_group(g, x.imag(), init.imag(), binary_op));

  return result;
#else
  throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                        "Group algorithms are not supported on host.");
#endif
}

/// Marray<Complex> specialization
template <typename Group, typename V, std::size_t N, typename T, std::size_t S,
          class BinaryOperation,
          typename = std::enable_if_t<
              sycl::is_group_v<std::decay_t<Group>> && is_gencomplex_v<V> &&
              is_gencomplex_v<T> &&
              cplex::detail::is_binary_op_supported_v<BinaryOperation>>>
sycl::marray<T, N> reduce_over_group(Group g, sycl::marray<V, N> x,
                                     sycl::marray<T, S> init,
                                     BinaryOperation binary_op) {
#ifdef __SYCL_DEVICE_ONLY__
  sycl::marray<T, N> result;

  cplex::detail::loop<N>([&](size_t s) {
    result[s] = reduce_over_group(g, x[s], init[s], binary_op);
  });

  return result;
#else
  throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                        "Group algorithms are not supported on host.");
#endif
}

/// Marray<Complex> and Complex specialization
template <typename Group, typename T, class BinaryOperation,
          typename = std::enable_if_t<
              sycl::is_group_v<std::decay_t<Group>> &&
              (is_gencomplex_v<T> || is_mgencomplex_v<T>)&&cplex::detail::
                  is_binary_op_supported_v<BinaryOperation>>>
T reduce_over_group(Group g, T x, BinaryOperation binary_op) {
#ifdef __SYCL_DEVICE_ONLY__
  auto init = cplex::detail::get_init<T, BinaryOperation>();

  return reduce_over_group(g, x, init, binary_op);
#else
  throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                        "Group algorithms are not supported on host.");
#endif
}

/* JOINT_REDUCE'S OVERLOADS */

/// Marray<Complex> and Complex specialization
template <typename Group, typename Ptr, typename T, class BinaryOperation,
          typename = std::enable_if_t<
              sycl::is_group_v<std::decay_t<Group>> &&
              sycl::detail::is_pointer<Ptr>::value &&
              (is_gencomplex_v<sycl::detail::remove_pointer_t<Ptr>> ||
               is_mgencomplex_v<sycl::detail::remove_pointer_t<
                   Ptr>>)&&(is_gencomplex_v<T> || is_mgencomplex_v<T>)&&cplex::
                  detail::is_binary_op_supported_v<BinaryOperation>>>
T joint_reduce(Group g, Ptr first, Ptr last, T init,
               BinaryOperation binary_op) {
#ifdef __SYCL_DEVICE_ONLY__
  auto partial = cplex::detail::get_init<T, BinaryOperation>();

  sycl::detail::for_each(
      g, first, last,
      [&](const typename sycl::detail::remove_pointer<Ptr>::type &x) {
        partial = binary_op(partial, x);
      });

  return reduce_over_group(g, partial, init, binary_op);
#else
  throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                        "Group algorithms are not supported on host.");
#endif
}

/// Marray<Complex> and Complex specialization
template <typename Group, typename Ptr, class BinaryOperation,
          typename = std::enable_if_t<
              sycl::is_group_v<std::decay_t<Group>> &&
              sycl::detail::is_pointer<Ptr>::value &&
              (is_gencomplex_v<sycl::detail::remove_pointer_t<Ptr>> ||
               is_mgencomplex_v<sycl::detail::remove_pointer_t<Ptr>>)&&cplex::
                  detail::is_binary_op_supported_v<BinaryOperation>>>
typename sycl::detail::remove_pointer_t<Ptr>
joint_reduce(Group g, Ptr first, Ptr last, BinaryOperation binary_op) {
#ifdef __SYCL_DEVICE_ONLY__
  using T = typename sycl::detail::remove_pointer_t<Ptr>;

  auto init = cplex::detail::get_init<T, BinaryOperation>();

  return joint_reduce(g, first, last, init, binary_op);
#else
  throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                        "Group algorithms are not supported on host.");
#endif
}

/* INCLUSIVE_SCAN_OVER_GROUP'S OVERLOADS */

/// Complex specialization
template <typename Group, typename V, class BinaryOperation, typename T,
          typename = std::enable_if_t<
              sycl::is_group_v<std::decay_t<Group>> && is_genfloat_v<V> &&
              is_genfloat_v<T> &&
              cplex::detail::is_binary_op_supported_v<BinaryOperation>>>
complex<T> inclusive_scan_over_group(Group g, complex<V> x,
                                     BinaryOperation binary_op,
                                     complex<T> init) {
#ifdef __SYCL_DEVICE_ONLY__
  complex<T> result;

  result.real(
      sycl::inclusive_scan_over_group(g, x.real(), binary_op, init.real()));
  result.imag(
      sycl::inclusive_scan_over_group(g, x.imag(), binary_op, init.imag()));

  return result;
#else
  throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                        "Group algorithms are not supported on host.");
#endif
}

/// Marray<Complex> specialization
template <typename Group, typename V, std::size_t N, class BinaryOperation,
          typename T, std::size_t S,
          typename = std::enable_if_t<
              sycl::is_group_v<std::decay_t<Group>> && is_gencomplex_v<V> &&
              is_gencomplex_v<T> &&
              cplex::detail::is_binary_op_supported_v<BinaryOperation>>>
sycl::marray<T, N> inclusive_scan_over_group(Group g, sycl::marray<V, N> x,
                                             BinaryOperation binary_op,
                                             sycl::marray<T, S> init) {
#ifdef __SYCL_DEVICE_ONLY__
  sycl::marray<T, N> result;

  cplex::detail::loop<N>([&](size_t s) {
    result[s] = inclusive_scan_over_group(g, x[s], binary_op, init[s]);
  });

  return result;
#else
  throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                        "Group algorithms are not supported on host.");
#endif
}

/// Marray<Complex> and Complex specialization
template <typename Group, typename T, class BinaryOperation,
          typename = std::enable_if_t<
              sycl::is_group_v<std::decay_t<Group>> &&
              (is_gencomplex_v<T> || is_mgencomplex_v<T>)&&cplex::detail::
                  is_binary_op_supported_v<BinaryOperation>>>
T inclusive_scan_over_group(Group g, T x, BinaryOperation binary_op) {
#ifdef __SYCL_DEVICE_ONLY__
  auto init = cplex::detail::get_init<T, BinaryOperation>();

  return inclusive_scan_over_group(g, x, binary_op, init);
#else
  throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                        "Group algorithms are not supported on host.");
#endif
}

/* JOINT_INCLUSIVE_SCAN'S OVERLOADS */

/// Complex specialization
template <typename Group, typename InPtr, typename OutPtr,
          class BinaryOperation, typename T,
          typename = std::enable_if_t<
              sycl::is_group_v<std::decay_t<Group>> &&
              sycl::detail::is_pointer<InPtr>::value &&
              sycl::detail::is_pointer<OutPtr>::value &&
              (is_gencomplex_v<sycl::detail::remove_pointer_t<InPtr>> ||
               is_mgencomplex_v<sycl::detail::remove_pointer_t<
                   InPtr>>)&&(is_gencomplex_v<sycl::detail::
                                                  remove_pointer_t<OutPtr>> ||
                              is_mgencomplex_v<sycl::detail::remove_pointer_t<
                                  OutPtr>>)&&(is_gencomplex_v<T> ||
                                              is_mgencomplex_v<T>)&&cplex::
                  detail::is_binary_op_supported_v<BinaryOperation>>>
OutPtr joint_inclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                            BinaryOperation binary_op, T init) {
#ifdef __SYCL_DEVICE_ONLY__
  std::ptrdiff_t offset = g.get_local_linear_id();
  std::ptrdiff_t stride = g.get_local_linear_range();
  std::ptrdiff_t N = last - first;

  auto roundup = [=](const std::ptrdiff_t &v,
                     const std::ptrdiff_t &divisor) -> std::ptrdiff_t {
    return ((v + divisor - 1) / divisor) * divisor;
  };

  typename std::remove_const_t<typename sycl::detail::remove_pointer_t<InPtr>>
      x;
  typename sycl::detail::remove_pointer_t<OutPtr> carry = init;

  for (std::ptrdiff_t chunk = 0; chunk < roundup(N, stride); chunk += stride) {
    std::ptrdiff_t i = chunk + offset;

    if (i < N)
      x = first[i];

    typename sycl::detail::remove_pointer_t<OutPtr> out =
        inclusive_scan_over_group(g, x, binary_op, carry);

    if (i < N)
      result[i] = out;

    carry = sycl::group_broadcast(g, out, stride - 1);
  }
  return result + N;
#else
  throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                        "Group algorithms are not supported on host.");
#endif
}

/// Complex specialization
template <
    typename Group, typename InPtr, typename OutPtr, class BinaryOperation,
    typename = std::enable_if_t<
        sycl::is_group_v<std::decay_t<Group>> &&
        sycl::detail::is_pointer<InPtr>::value &&
        sycl::detail::is_pointer<OutPtr>::value &&
        (is_gencomplex_v<sycl::detail::remove_pointer_t<InPtr>> ||
         is_mgencomplex_v<sycl::detail::remove_pointer_t<
             InPtr>>)&&(is_gencomplex_v<sycl::detail::
                                            remove_pointer_t<OutPtr>> ||
                        is_mgencomplex_v<
                            sycl::detail::remove_pointer_t<OutPtr>>)&&cplex::
            detail::is_binary_op_supported_v<BinaryOperation>>>
OutPtr joint_inclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                            BinaryOperation binary_op) {
#ifdef __SYCL_DEVICE_ONLY__
  using T = typename sycl::detail::remove_pointer_t<InPtr>;

  auto init = cplex::detail::get_init<T, BinaryOperation>();

  return joint_inclusive_scan(g, first, last, result, binary_op, init);
#else
  throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                        "Group algorithms are not supported on host.");
#endif
}

/* EXCLUSIVE_SCAN_OVER_GROUP'S OVERLOADS */

/// Complex specialization
template <typename Group, typename V, typename T, class BinaryOperation,
          typename = std::enable_if_t<
              sycl::is_group_v<std::decay_t<Group>> && is_genfloat_v<V> &&
              is_genfloat_v<T> &&
              cplex::detail::is_binary_op_supported_v<BinaryOperation>>>
complex<T> exclusive_scan_over_group(Group g, complex<V> x, complex<T> init,
                                     BinaryOperation binary_op) {
#ifdef __SYCL_DEVICE_ONLY__
  complex<T> result;

  result.real(
      sycl::exclusive_scan_over_group(g, x.real(), init.real(), binary_op));
  result.imag(
      sycl::exclusive_scan_over_group(g, x.imag(), init.imag(), binary_op));

  return result;
#else
  throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                        "Group algorithms are not supported on host.");
#endif
}

/// Marray<Complex> specialization
template <typename Group, typename V, std::size_t N, typename T, std::size_t S,
          class BinaryOperation,
          typename = std::enable_if_t<
              sycl::is_group_v<std::decay_t<Group>> && is_gencomplex_v<V> &&
              is_gencomplex_v<T> &&
              cplex::detail::is_binary_op_supported_v<BinaryOperation>>>
sycl::marray<T, N> exclusive_scan_over_group(Group g, sycl::marray<V, N> x,
                                             sycl::marray<T, S> init,
                                             BinaryOperation binary_op) {
#ifdef __SYCL_DEVICE_ONLY__
  sycl::marray<T, N> result;

  cplex::detail::loop<N>([&](size_t s) {
    result[s] = exclusive_scan_over_group(g, x[s], init[s], binary_op);
  });

  return result;
#else
  throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                        "Group algorithms are not supported on host.");
#endif
}

/// Marray<Complex> and Complex specialization
template <typename Group, typename T, class BinaryOperation,
          typename = std::enable_if_t<
              sycl::is_group_v<std::decay_t<Group>> &&
              (is_gencomplex_v<T> || is_mgencomplex_v<T>)&&cplex::detail::
                  is_binary_op_supported_v<BinaryOperation>>>
T exclusive_scan_over_group(Group g, T x, BinaryOperation binary_op) {
#ifdef __SYCL_DEVICE_ONLY__
  auto init = cplex::detail::get_init<T, BinaryOperation>();

  return exclusive_scan_over_group(g, x, init, binary_op);
#else
  throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                        "Group algorithms are not supported on host.");
#endif
}

/* JOINT_EXCLUSIVE_SCAN'S OVERLOADS */

/// Complex specialization
template <typename Group, typename InPtr, typename OutPtr, typename T,
          class BinaryOperation,
          typename = std::enable_if_t<
              sycl::is_group_v<std::decay_t<Group>> &&
              sycl::detail::is_pointer<InPtr>::value &&
              sycl::detail::is_pointer<OutPtr>::value &&
              (is_gencomplex_v<sycl::detail::remove_pointer_t<InPtr>> ||
               is_mgencomplex_v<sycl::detail::remove_pointer_t<
                   InPtr>>)&&(is_gencomplex_v<sycl::detail::
                                                  remove_pointer_t<OutPtr>> ||
                              is_mgencomplex_v<sycl::detail::remove_pointer_t<
                                  OutPtr>>)&&(is_gencomplex_v<T> ||
                                              is_mgencomplex_v<T>)&&
              //
              cplex::detail::is_binary_op_supported_v<BinaryOperation>>>
OutPtr joint_exclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                            T init, BinaryOperation binary_op) {
#ifdef __SYCL_DEVICE_ONLY__
  std::ptrdiff_t offset = g.get_local_linear_id();
  std::ptrdiff_t stride = g.get_local_linear_range();
  std::ptrdiff_t N = last - first;

  auto roundup = [=](const std::ptrdiff_t &v,
                     const std::ptrdiff_t &divisor) -> std::ptrdiff_t {
    return ((v + divisor - 1) / divisor) * divisor;
  };

  typename std::remove_const_t<typename sycl::detail::remove_pointer_t<InPtr>>
      x;
  typename sycl::detail::remove_pointer_t<OutPtr> carry = init;

  for (std::ptrdiff_t chunk = 0; chunk < roundup(N, stride); chunk += stride) {
    std::ptrdiff_t i = chunk + offset;
    if (i < N)
      x = first[i];

    typename sycl::detail::remove_pointer_t<OutPtr> out =
        exclusive_scan_over_group(g, x, carry, binary_op);

    if (i < N)
      result[i] = out;

    carry = sycl::group_broadcast(g, binary_op(out, x), stride - 1);
  }
  return result + N;
#else
  throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                        "Group algorithms are not supported on host.");
#endif
}

/// Complex specialization
template <
    typename Group, typename InPtr, typename OutPtr, class BinaryOperation,
    typename = std::enable_if_t<
        sycl::is_group_v<std::decay_t<Group>> &&
        sycl::detail::is_pointer<InPtr>::value &&
        sycl::detail::is_pointer<OutPtr>::value &&
        (is_gencomplex_v<sycl::detail::remove_pointer_t<InPtr>> ||
         is_mgencomplex_v<sycl::detail::remove_pointer_t<
             InPtr>>)&&(is_gencomplex_v<sycl::detail::
                                            remove_pointer_t<OutPtr>> ||
                        is_mgencomplex_v<
                            sycl::detail::remove_pointer_t<OutPtr>>)&&cplex::
            detail::is_binary_op_supported_v<BinaryOperation>>>
OutPtr joint_exclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                            BinaryOperation binary_op) {
#ifdef __SYCL_DEVICE_ONLY__
  using T = typename sycl::detail::remove_pointer_t<InPtr>;

  auto init = cplex::detail::get_init<T, BinaryOperation>();

  return joint_exclusive_scan(g, first, last, result, init, binary_op);
#else
  throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                        "Group algorithms are not supported on host.");
#endif
}

_SYCL_EXT_CPLX_END_NAMESPACE_STD

#undef _SYCL_MARRAY_BEGIN_NAMESPACE
#undef _SYCL_MARRAY_END_NAMESPACE

#undef _SYCL_EXT_CPLX_BEGIN_NAMESPACE_STD
#undef _SYCL_EXT_CPLX_END_NAMESPACE_STD
#undef _SYCL_EXT_CPLX_INLINE_VISIBILITY

#endif // _SYCL_EXT_CPLX_COMPLEX
