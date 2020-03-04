#pragma once

#include "prologue.hpp"

#include <type_traits>
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
using has_sender_traits = detail::conjunction<has_value_types<T>, has_error_types<T>, has_sends_done<T>>;


template<class S, class Enable = void>
struct sender_traits_base {};

// If S is a reference, strip the reference and implement with sender_traits
template<class S>
struct sender_traits_base<S,
  typename std::enable_if<
    !std::is_same<
      S, remove_cvref_t<S>
    >::value
  >::type
> : sender_traits<remove_cvref_t<S>>
{};


template<class S>
struct sender_traits_base<S,
  typename std::enable_if<
    std::is_same<S, remove_cvref_t<S>>::value and
    is_sender<S>::value and
    has_sender_traits<S>::value
  >::type
>
{
  template<template<class...> class Tuple, template<class...> class Variant>
  using value_types = typename S::template value_types<Tuple, Variant>;

  template<template<class...> class Variant>
  using error_types = typename S::template error_types<Variant>;

  static constexpr bool sends_done = S::sends_done;
};


} // end detail


template<class S>
struct sender_traits : detail::sender_traits_base<S> {};


P0443_NAMESPACE_CLOSE_BRACE


#include "epilogue.hpp"

