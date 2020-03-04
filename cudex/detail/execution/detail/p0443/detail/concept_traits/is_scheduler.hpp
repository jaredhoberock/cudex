#pragma once

#include "../prologue.hpp"

#include <type_traits>
#include "../customization_points/schedule.hpp"
#include "../type_traits/is_detected.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class S>
using schedule_t = decltype(P0443_NAMESPACE::schedule(std::declval<S>()));


template<class S>
using is_scheduler = is_detected<schedule_t, S>;


} // end detail


namespace ext
{


template<class S>
using is_scheduler = detail::is_scheduler<S>;
  

} // end ext


P0443_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

