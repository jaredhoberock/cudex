#pragma once

#include "../prologue.hpp"

#include <type_traits>
#include "../customization_points/execute.hpp"
#include "../type_traits/is_detected.hpp"
#include "../type_traits/is_equality_comparable.hpp"
#include "../type_traits/is_invocable.hpp"
#include "../type_traits/standard_traits.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class E, class F>
using execute_t = decltype(P0443_NAMESPACE::execute(std::declval<E>(), std::declval<F>()));


template<class E, class F>
using is_executor_of = std::integral_constant<
  bool,
  is_invocable<F>::value and
  std::is_nothrow_copy_constructible<E>::value and
  std::is_nothrow_destructible<E>::value and
  is_equality_comparable<E>::value and
  is_detected<detail::execute_t, E, F>::value
>;


} // end detail


namespace ext
{


template<class E, class F>
using is_executor_of = detail::is_executor_of<E,F>;


} // end ext


P0443_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

