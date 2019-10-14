
// ======================================================================
// gtl.h
//
// generic template metaprogramming pieces
//
// Copyright (C) 2019 Kai Germaschewski

#ifndef GTENSOR_GTL_H
#define GTENSOR_GTL_H

#include "macros.h"

namespace gt
{

// ======================================================================
// disjunction

template <class...>
struct disjunction : std::false_type
{};
template <class B1>
struct disjunction<B1> : B1
{};
template <class B1, class... Bn>
struct disjunction<B1, Bn...>
  : std::conditional_t<bool(B1::value), B1, disjunction<Bn...>>
{};

// ======================================================================
// assert_is_same
//
// checks whether type matches expected type -- for debugging

template <typename Exp, typename T>
void assert_is_same()
{
  static_assert(std::is_same<Exp, T>::value, "not expected type");
};

// ======================================================================
// debug_type

template <typename T>
struct debug_type
{
  static_assert(!std::is_same<T, T>::value, "debug_type");
};

template <typename... Ts>
struct always_false : std::false_type
{};

template <typename... Ts>
void debug_types()
{
  static_assert(always_false<Ts...>::value, "debug_type");
};

} // namespace gt

#endif
