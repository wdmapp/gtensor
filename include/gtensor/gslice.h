
#ifndef GTENSOR_GSLICE_H
#define GTENSOR_GSLICE_H

#include <limits>

namespace gt
{

// ======================================================================
// gslice

struct gnone
{};

namespace placeholders
{
constexpr gnone _{};
}

class gslice
{
public:
  enum Type
  {
    ALL,
    SLICE,
    VALUE,
    NEWAXIS,
  };

  static const int none = std::numeric_limits<int>::min();

  gslice(Type type, int start = none, int stop = none)
    : type_(type), start_(start), stop_(stop)
  {}

  gslice(Type type, int start, gnone) : type_(type), start_(start), stop_(none)
  {}

  gslice(Type type, gnone, int stop) : type_(type), start_(none), stop_(stop) {}

  gslice(Type type, gnone, gnone) : type_(type), start_(none), stop_(none) {}

  gslice(int value) : type_(VALUE), start_(value) {}

  Type type() const { return type_; }

  int value() const
  {
    assert(type_ == VALUE);
    return start_;
  }
  int start() const
  {
    assert(type_ == SLICE);
    return start_;
  }
  int stop() const
  {
    assert(type_ == SLICE);
    return stop_;
  }

private:
  enum Type type_;
  int start_;
  int stop_;
};

template <typename T1, typename T2>
inline gslice slice(T1 start, T2 stop)
{
  return gslice(gslice::SLICE, start, stop);
}

inline gslice all()
{
  return gslice(gslice::ALL);
}

inline gslice newaxis()
{
  return gslice(gslice::NEWAXIS);
}

inline std::ostream& operator<<(std::ostream& os, const gslice& slice)
{
  if (slice.type() == gslice::ALL) {
    os << "all()";
  } else if (slice.type() == gslice::VALUE) {
    os << slice.value();
  } else if (slice.type() == gslice::SLICE) {
    os << slice.start() << ":" << slice.stop();
  }
  return os;
}

} // namespace gt

#endif
