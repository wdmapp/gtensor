#ifndef TEST_DEBUG_H
#define TEST_DEBUG_H

#ifdef GTENSOR_TEST_DEBUG

#include <iostream>

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
    std::cerr << #x << " type: " << typeid(x).name() << std::endl;             \
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

#endif // GTENSOR_TEST_DEBUG

#endif // TEST_DEBUG_H
