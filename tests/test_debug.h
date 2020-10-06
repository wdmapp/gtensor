#ifndef TEST_DEBUG_H
#define TEST_DEBUG_H

#ifdef GTENSOR_TEST_DEBUG

#include <iostream>

// See https://stackoverflow.com/a/35943472

struct string_view
{
  char const* data;
  std::size_t size;
};

inline std::ostream& operator<<(std::ostream& o, string_view const& s)
{
  return o.write(s.data, s.size);
}

template <class T>
constexpr string_view get_type_name()
{
  char const* p = __PRETTY_FUNCTION__;
  while (*p++ != '=')
    ;
  for (; *p == ' '; ++p)
    ;
  char const* p2 = p;
  int count = 1;
  for (;; ++p2) {
    switch (*p2) {
      case '[': ++count; break;
      case ']':
        --count;
        if (!count)
          return {p, std::size_t(p2 - p)};
    }
  }
  return {};
}

#define GT_DEBUG_PRINTLN(x)                                                    \
  do {                                                                         \
    std::cerr << x << std::endl;                                               \
  } while (0)

#define GT_DEBUG_VAR(x)                                                        \
  do {                                                                         \
    std::cerr << #x << " = " << x << std::endl;                                \
  } while (0)

#define GT_DEBUG_TYPE(x)                                                       \
  do {                                                                         \
    std::cerr << #x << " type: " << get_type_name<decltype(x)>() << std::endl; \
  } while (0)

#define GT_DEBUG_TYPE_NAME(t)                                                  \
  do {                                                                         \
    std::cerr << #t << " = " << get_type_name<t>() << std::endl;               \
  } while (0)

#else // not defined GTENSOR_TEST_DEBUG

#define GT_DEBUG_PRINTLN(x)                                                    \
  do {                                                                         \
  } while (0)

#define GT_DEBUG_VAR(x)                                                        \
  do {                                                                         \
  } while (0)

#define GT_DEBUG_TYPE(x)                                                       \
  do {                                                                         \
  } while (0)

#define GT_DEBUG_TYPE_NAME(x)                                                  \
  do {                                                                         \
  } while (0)

#endif // GTENSOR_TEST_DEBUG

#endif // TEST_DEBUG_H
