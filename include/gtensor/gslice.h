
#ifndef GTENSOR_GSLICE_H
#define GTENSOR_GSLICE_H

#include <limits>

namespace gt
{

struct gnewaxis
{};

struct gnone
{};

constexpr gnone none;

// ======================================================================
// gslice

struct gslice
{
  static const int none = std::numeric_limits<int>::min();

  gslice() = default;
  gslice(int start, int stop, int step) : start(start), stop(stop), step(step)
  {}
  gslice(int start, int stop, gnone) : start(start), stop(stop) {}
  gslice(int start, gnone, int step) : start(start), step(step) {}
  gslice(gnone, int stop, int step) : stop(stop), step(step) {}
  gslice(int start, gnone, gnone) : start(start) {}
  gslice(gnone, int stop, gnone) : stop(stop) {}
  gslice(gnone, gnone, int step) : step(step) {}
  gslice(gnone, gnone, gnone) {}

  int start = none;
  int stop = none;
  int step = none;
};

// ======================================================================
// detail

namespace detail
{

template <typename U, typename... Ts>
struct count_impl;

template <typename U>
struct count_impl<U>
{
  static constexpr std::size_t value = 0;
};

template <typename U, typename T, typename... Ts>
struct count_impl<U, T, Ts...>
{
  static constexpr std::size_t value =
    std::is_convertible<T, U>::value + count_impl<U, Ts...>::value;
};

template <typename U, typename... Ts>
constexpr std::size_t count_convertible()
{
  return count_impl<U, Ts...>::value;
}

}; // namespace detail

// FIXME, poor man's std::variant
class gdesc
{
public:
  enum Type
  {
    ALL,
    SLICE,
    VALUE,
    NEWAXIS,
  };

  gdesc(gnewaxis) : type_(NEWAXIS) {}
  gdesc(int value) : type_(VALUE), value_(value) {}
  gdesc(const gslice& slice) : type_(SLICE), slice_(slice) {}

  Type type() const { return type_; }

  int value() const
  {
    assert(type_ == VALUE);
    return value_;
  }

  gslice slice() const
  {
    assert(type_ == SLICE);
    return slice_;
  }

private:
  enum Type type_;
  union
  {
    int value_;
    gslice slice_;
  };
};

template <typename T1, typename T2>
inline gslice slice(T1 start, T2 stop)
{
  return gslice(start, stop, gt::none);
}

template <typename T1, typename T2, typename T3>
inline gslice slice(T1 start, T2 stop, T3 step)
{
  return gslice(start, stop, step);
}

constexpr gslice all{};
constexpr gnewaxis newaxis{};

inline std::ostream& operator<<(std::ostream& os, const gdesc& desc)
{
  if (desc.type() == gdesc::ALL) {
    os << ":";
  } else if (desc.type() == gdesc::NEWAXIS) {
    os << "newaxis";
  } else if (desc.type() == gdesc::VALUE) {
    os << desc.value();
  } else if (desc.type() == gdesc::SLICE) {
    os << desc.slice().start << ":" << desc.slice().stop << ":"
       << desc.slice().step;
  }
  return os;
}

namespace placeholders
{
constexpr gslice _all;
constexpr gnewaxis _newaxis;

template <typename... Ts>
inline gslice _s(Ts... ts)
{
  return slice(std::forward<Ts>(ts)...);
}

constexpr gnone _;
} // namespace placeholders

} // namespace gt

#endif
