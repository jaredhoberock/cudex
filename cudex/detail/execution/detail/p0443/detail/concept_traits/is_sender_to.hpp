#pragma once

#include "../prologue.hpp"

#include <type_traits>
#include "../type_traits/is_detected.hpp"
#include "../type_traits/standard_traits.hpp"
#include "../customization_points/submit.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class S, class R>
using submit_t = decltype(P0443_NAMESPACE::submit(std::declval<S>(), std::declval<R>()));


template<class S, class R>
using is_sender_to = std::integral_constant<
  bool,
  is_detected<submit_t, S, R>::value and std::is_move_constructible<remove_cvref_t<S>>::value
>;


} // end detail


namespace ext
{


template<class S, class R>
using is_sender_to = detail::is_sender_to<S,R>;
  

} // end ext


P0443_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

