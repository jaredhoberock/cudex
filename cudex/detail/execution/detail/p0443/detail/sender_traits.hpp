#pragma once

#include "prologue.hpp"

#include <type_traits>
#include "sender_base.hpp"
#include "type_traits/standard_traits.hpp"


P0443_NAMESPACE_OPEN_BRACE


template<class S>
struct sender_traits;


namespace detail
{


template<class T>
struct has_value_types
{
  template<class...>
  struct template_template_parameter {};

  template<class U = T,
           class = typename U::template value_types<template_template_parameter, template_template_parameter>
          >
  constexpr static bool test(int)
  {
    return true;
  }

  template<class>
  constexpr static bool test(...)
  {
    return false;
  }

  constexpr static bool value = test<T>(0);
};


template<class T>
struct has_error_types
{
  template<class...>
  struct template_template_parameter {};

  template<class U = T,
           class = typename U::template error_types<template_template_parameter>
          >
  constexpr static bool test(int)
  {
    return true;
  }

  template<class>
  constexpr static bool test(...)
  {
    return false;
  }

  constexpr static bool value = test<T>(0);
};


template<class T>
struct has_sends_done
{
  template<class U = T,
           bool = U::sends_done
          >
  constexpr static bool test(int)
  {
    return true;
  }

  template<class>
  constexpr static bool test(...)
  {
    return false;
  }

  constexpr static bool value = test<T>(0);
};


template<class T>
using has_sender_types = detail::conjunction<has_value_types<T>, has_error_types<T>, has_sends_done<T>>;


template<class S, class Enable = void>
struct sender_traits_base
{
  using __unspecialized = void;
};


// If S has sender types, use them
template<class S>
struct sender_traits_base<
  S,
  typename std::enable_if<
    has_sender_types<S>::value
  >::type
>
{
  template<template<class...> class Tuple, template<class...> class Variant>
  using value_types = typename S::template value_types<Tuple, Variant>;

  template<template<class...> class Variant>
  using error_types = typename S::template error_types<Variant>;

  static constexpr bool sends_done = S::sends_done;
};


// XXX Otherwise, if executor-of-impl<S,as-invocable<void-receiver, S>> is true, then sender-traits-base is equivalent to
// XXX TODO


// Otherwise, if S does not have sender types, and S is derived from sender_base
template<class Derived, class Base>
using is_derived_from = std::integral_constant<
  bool,
  std::is_base_of<Base,Derived>::value and
  std::is_convertible<const volatile Derived*, const volatile Base*>::value
>;

template<class S>
struct sender_traits_base<
  S,
  typename std::enable_if<
    !has_sender_types<S>::value and
    is_derived_from<S, sender_base>::value
  >::type
>
{
  // empty
};


} // end detail


template<class S>
struct sender_traits : detail::sender_traits_base<detail::remove_cvref_t<S>> {};


P0443_NAMESPACE_CLOSE_BRACE


#include "epilogue.hpp"

