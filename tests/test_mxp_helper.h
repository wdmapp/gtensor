#ifndef GTENSOR_TEST_MXP_HELPER_H
#define GTENSOR_TEST_MXP_HELPER_H

#include <cstdint>
#include <type_traits>

// -------------------------------------------------------------------------- //

// auxiliary to loop over compile time <bits>
// in test_mxp_truncated_mantissa_t.cxx

template <std::uint16_t From, std::uint16_t To, typename Task>
struct Loop
{
  template <typename... Args>
  static std::enable_if_t<From <= To> Run(Args&&... args)
  {
    Task::template Iteration<From>(std::forward<Args>(args)...);
    Loop<From + 1, To, Task>::Run(args...);
  }
};

template <std::uint16_t Final, typename Task>
struct Loop<Final, Final, Task>
{
  template <typename... Args>
  static void Run(Args&&... args)
  {
    Task::template Iteration<Final>(std::forward<Args>(args)...);
  }
};

// -------------------------------------------------------------------------- //

#endif // GTENSOR_TEST_MXP_HELPER_H
