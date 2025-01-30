#ifndef GTENSOR_TEST_MXT_HELPER_H
#define GTENSOR_TEST_MXT_HELPER_H

#include <type_traits>

// -------------------------------------------------------------------------- //

template <std::uint8_t From, std::uint8_t To, typename Task>
struct Loop
{
  template <typename... Args>
  static std::enable_if_t<From <= To> Run(Args&&... args)
  {
    Task::template Iteration<From>(std::forward<Args>(args)...);
    Loop<From + 1, To, Task>::Run(args...);
  }
};

template <std::uint8_t Final, typename Task>
struct Loop<Final, Final, Task>
{
  template <typename... Args>
  static void Run(Args&&... args)
  {
    Task::template Iteration<Final>(std::forward<Args>(args)...);
  }
};

// -------------------------------------------------------------------------- //

#endif // GTENSOR_TEST_MXT_HELPER_H
