#ifndef GTENSOR_COMPLEX_FLOAT16T_H
#define GTENSOR_COMPLEX_FLOAT16T_H

#include <iostream>

#include "complex.h"
#include "float16_t.h"
#include "macros.h"

namespace gt
{

// ======================================================================
// complex_float16_t
// ... adapted from the C++ <complex> header,
// see e.g., https://en.cppreference.com/w/cpp/header/complex [2023/10/17]

class complex_float16_t;

// operators:
GT_INLINE complex_float16_t operator+(const complex_float16_t&,
                                      const complex_float16_t&);
GT_INLINE complex_float16_t operator+(const complex_float16_t&,
                                      const float16_t&);
GT_INLINE complex_float16_t operator+(const float16_t&,
                                      const complex_float16_t&);

GT_INLINE complex_float16_t operator-(const complex_float16_t&,
                                      const complex_float16_t&);
GT_INLINE complex_float16_t operator-(const complex_float16_t&,
                                      const float16_t&);
GT_INLINE complex_float16_t operator-(const float16_t&,
                                      const complex_float16_t&);

GT_INLINE complex_float16_t operator*(const complex_float16_t&,
                                      const complex_float16_t&);
GT_INLINE complex_float16_t operator*(const complex_float16_t&,
                                      const float16_t&);
GT_INLINE complex_float16_t operator*(const float16_t&,
                                      const complex_float16_t&);

GT_INLINE complex_float16_t operator/(const complex_float16_t&,
                                      const complex_float16_t&);
GT_INLINE complex_float16_t operator/(const complex_float16_t&,
                                      const float16_t&);
GT_INLINE complex_float16_t operator/(const float16_t&,
                                      const complex_float16_t&);

GT_INLINE complex_float16_t operator+(const complex_float16_t&);
GT_INLINE complex_float16_t operator-(const complex_float16_t&);

GT_INLINE bool operator==(const complex_float16_t&, const complex_float16_t&);
GT_INLINE bool operator==(const complex_float16_t&, const float16_t&);
GT_INLINE bool operator==(const float16_t&, const complex_float16_t&);

GT_INLINE bool operator!=(const complex_float16_t&, const complex_float16_t&);
GT_INLINE bool operator!=(const complex_float16_t&, const float16_t&);
GT_INLINE bool operator!=(const float16_t&, const complex_float16_t&);

template <class CharT, class Traits>
std::basic_istream<CharT, Traits>& operator>>(
  std::basic_istream<CharT, Traits>&, complex_float16_t&);

template <class CharT, class Traits>
std::basic_ostream<CharT, Traits>& operator<<(
  std::basic_ostream<CharT, Traits>&, const complex_float16_t&);

// values:
GT_INLINE float16_t real(const complex_float16_t&);
GT_INLINE float16_t imag(const complex_float16_t&);

GT_INLINE float16_t abs(const complex_float16_t&);
GT_INLINE float16_t norm(const complex_float16_t&);

GT_INLINE complex_float16_t conj(const complex_float16_t&);

// values = delete [NOT IMPLEMENTED]
float16_t arg(const complex_float16_t&) = delete;
complex_float16_t proj(const complex_float16_t&) = delete;
complex_float16_t polar(const float16_t&, const float16_t& = 0) = delete;

// transcendentals = delete [NOT IMPLEMENTED]
complex_float16_t acos(const complex_float16_t&) = delete;
complex_float16_t asin(const complex_float16_t&) = delete;
complex_float16_t atan(const complex_float16_t&) = delete;

complex_float16_t acosh(const complex_float16_t&) = delete;
complex_float16_t asinh(const complex_float16_t&) = delete;
complex_float16_t atanh(const complex_float16_t&) = delete;

complex_float16_t cos(const complex_float16_t&) = delete;
complex_float16_t cosh(const complex_float16_t&) = delete;
complex_float16_t exp(const complex_float16_t&) = delete;
complex_float16_t log(const complex_float16_t&) = delete;
complex_float16_t log10(const complex_float16_t&) = delete;

complex_float16_t pow(const complex_float16_t&, const float16_t&) = delete;
complex_float16_t pow(const complex_float16_t&,
                      const complex_float16_t&) = delete;
complex_float16_t pow(const float16_t&, const complex_float16_t&) = delete;

complex_float16_t sin(const complex_float16_t&) = delete;
complex_float16_t sinh(const complex_float16_t&) = delete;
complex_float16_t sqrt(const complex_float16_t&) = delete;
complex_float16_t tan(const complex_float16_t&) = delete;
complex_float16_t tanh(const complex_float16_t&) = delete;

class complex_float16_t
{
public:
  typedef float16_t value_type;
  GT_INLINE complex_float16_t(const float16_t& re = float16_t(),
                              const float16_t& im = float16_t())
    : _real(re), _imag(im)
  {}
  complex_float16_t(const complex_float16_t&) = default;
  template <class X>
  GT_INLINE explicit complex_float16_t(const complex<X>& z)
    : _real(z.real()), _imag(z.imag())
  {}

  GT_INLINE float16_t real() const { return _real; }
  GT_INLINE void real(float16_t re) { _real = re; }
  GT_INLINE float16_t imag() const { return _imag; }
  GT_INLINE void imag(float16_t im) { _imag = im; }

  GT_INLINE complex_float16_t& operator=(const float16_t& x)
  {
    _real = x;
    _imag = 0;
    return *this;
  }
  GT_INLINE complex_float16_t& operator+=(const float16_t& x)
  {
    _real += x;
    return *this;
  }
  GT_INLINE complex_float16_t& operator-=(const float16_t& x)
  {
    _real -= x;
    return *this;
  }
  GT_INLINE complex_float16_t& operator*=(const float16_t& x)
  {
    _real *= x;
    _imag *= x;
    return *this;
  }
  GT_INLINE complex_float16_t& operator/=(const float16_t& x)
  {
    _real /= x;
    _imag /= x;
    return *this;
  }

  complex_float16_t& operator=(const complex_float16_t&) = default;
  GT_INLINE complex_float16_t& operator+=(const complex_float16_t& z)
  {
    _real += z.real();
    _imag += z.imag();
    return *this;
  }
  GT_INLINE complex_float16_t& operator-=(const complex_float16_t& z)
  {
    _real -= z.real();
    _imag -= z.imag();
    return *this;
  }
  GT_INLINE complex_float16_t& operator*=(const complex_float16_t& z)
  {
    const auto retmp{_real};
    _real = _real * z.real() - _imag * z.imag();
    _imag = _imag * z.real() + retmp * z.imag();
    return *this;
  }
  GT_INLINE complex_float16_t& operator/=(const complex_float16_t& z)
  {
    auto z_alt = conj(z);
    z_alt /= norm(z);
    *this *= z_alt;
    return *this;
  }

  template <class X>
  GT_INLINE complex_float16_t& operator=(const complex<X>& z)
  {
    _real = z.real();
    _imag = z.imag();
    return *this;
  }
  template <class X>
  GT_INLINE complex_float16_t& operator+=(const complex<X>& z)
  {
    *this += complex_float16_t{z};
    return *this;
  }
  template <class X>
  GT_INLINE complex_float16_t& operator-=(const complex<X>& z)
  {
    *this -= complex_float16_t{z};
    return *this;
  }
  template <class X>
  GT_INLINE complex_float16_t& operator*=(const complex<X>& z)
  {
    *this *= complex_float16_t{z};
    return *this;
  }
  template <class X>
  GT_INLINE complex_float16_t& operator/=(const complex<X>& z)
  {
    *this /= complex_float16_t{z};
    return *this;
  }

private:
  float16_t _real;
  float16_t _imag;
};

// operators:
GT_INLINE complex_float16_t operator+(const complex_float16_t& lhs,
                                      const complex_float16_t& rhs)
{
  complex_float16_t result{lhs};
  result += rhs;
  return result;
}

GT_INLINE complex_float16_t operator+(const complex_float16_t& lhs,
                                      const float16_t& rhs)
{
  complex_float16_t result{lhs};
  result += rhs;
  return result;
}
GT_INLINE complex_float16_t operator+(const float16_t& lhs,
                                      const complex_float16_t& rhs)
{
  complex_float16_t result{lhs};
  result += rhs;
  return result;
}

GT_INLINE complex_float16_t operator-(const complex_float16_t& lhs,
                                      const complex_float16_t& rhs)
{
  complex_float16_t result{lhs};
  result -= rhs;
  return result;
}
GT_INLINE complex_float16_t operator-(const complex_float16_t& lhs,
                                      const float16_t& rhs)
{
  complex_float16_t result{lhs};
  result -= rhs;
  return result;
}
GT_INLINE complex_float16_t operator-(const float16_t& lhs,
                                      const complex_float16_t& rhs)
{
  complex_float16_t result{lhs};
  result -= rhs;
  return result;
}

GT_INLINE complex_float16_t operator*(const complex_float16_t& lhs,
                                      const complex_float16_t& rhs)
{
  complex_float16_t result{lhs};
  result *= rhs;
  return result;
}
GT_INLINE complex_float16_t operator*(const complex_float16_t& lhs,
                                      const float16_t& rhs)
{
  complex_float16_t result{lhs};
  result *= rhs;
  return result;
}
GT_INLINE complex_float16_t operator*(const float16_t& lhs,
                                      const complex_float16_t& rhs)
{
  complex_float16_t result{lhs};
  result *= rhs;
  return result;
}

GT_INLINE complex_float16_t operator/(const complex_float16_t& lhs,
                                      const complex_float16_t& rhs)
{
  complex_float16_t result{lhs};
  result /= rhs;
  return result;
}
GT_INLINE complex_float16_t operator/(const complex_float16_t& lhs,
                                      const float16_t& rhs)
{
  complex_float16_t result{lhs};
  result /= rhs;
  return result;
}
GT_INLINE complex_float16_t operator/(const float16_t& lhs,
                                      const complex_float16_t& rhs)
{
  complex_float16_t result{lhs};
  result /= rhs;
  return result;
}

GT_INLINE complex_float16_t operator+(const complex_float16_t& z) { return z; }
GT_INLINE complex_float16_t operator-(const complex_float16_t& z)
{
  return complex_float16_t{-z.real(), -z.imag()};
}

GT_INLINE bool operator==(const complex_float16_t& lhs,
                          const complex_float16_t& rhs)
{
  return lhs.real() == rhs.real() && lhs.imag() == rhs.imag();
}
GT_INLINE bool operator==(const complex_float16_t& lhs, const float16_t& rhs)
{
  return lhs.real() == rhs && lhs.imag() == 0;
}
GT_INLINE bool operator==(const float16_t& lhs, const complex_float16_t& rhs)
{
  return lhs == rhs.real() && 0 == rhs.imag();
}

GT_INLINE bool operator!=(const complex_float16_t& lhs,
                          const complex_float16_t& rhs)
{
  return lhs.real() != rhs.real() || lhs.imag() != rhs.imag();
}
GT_INLINE bool operator!=(const complex_float16_t& lhs, const float16_t& rhs)
{
  return lhs.real() != rhs || lhs.imag() != 0;
}
GT_INLINE bool operator!=(const float16_t& lhs, const complex_float16_t& rhs)
{
  return lhs != rhs.real() || 0 != rhs.imag();
}

template <class CharT, class Traits>
std::basic_istream<CharT, Traits>& operator>>(
  std::basic_istream<CharT, Traits>& s, complex_float16_t& z)
{
  complex<float> w;
  s >> w;
  z = w;
  return s;
}

template <class CharT, class Traits>
std::basic_ostream<CharT, Traits>& operator<<(
  std::basic_ostream<CharT, Traits>& s, const complex_float16_t& z)
{
  return s << "(" << z.real() << ", " << z.imag() << ")";
}

// values:
GT_INLINE float16_t real(const complex_float16_t& z) { return z.real(); }
GT_INLINE float16_t imag(const complex_float16_t& z) { return z.imag(); }

GT_INLINE float16_t abs(const complex_float16_t& z)
{
  auto abs2 = norm(z);
  return sqrt(abs2);
}
GT_INLINE float16_t norm(const complex_float16_t& z)
{
  return z.real() * z.real() + z.imag() * z.imag();
}

GT_INLINE complex_float16_t conj(const complex_float16_t& z)
{
  return complex_float16_t{z.real(), -z.imag()};
}

} // namespace gt

#endif // GTENSOR_COMPLEX_FLOAT16T_H
