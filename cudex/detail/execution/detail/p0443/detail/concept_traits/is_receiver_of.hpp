#pragma once

#include "../prologue.hpp"

#include <type_traits>
#include "../concept_traits/is_receiver.hpp"
#include "../customization_points/set_value.hpp"
#include "../type_traits/is_detected.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class R, class... Args>
using set_value_t = decltype(P0443_NAMESPACE::set_value(std::declval<R>(), std::declval<Args>()...));


template<class R, class... Args>
struct is_receiver_of : std::integral_constant<
  bool,
  is_receiver<R>::value and
  is_detected<set_value_t, R, Args...>::value
>
{};


// specialization for receiver of void
template<class R>
struct is_receiver_of<R,void> : std::integral_constant<
  bool,
  is_receiver<R>::value and
  is_detected<set_value_t, R>::value
>
{};


} // end detail


namespace ext
{


template<class R, class... Args>
using is_receiver_of = detail::is_receiver_of<R,Args...>;


} // end ext


P0443_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

