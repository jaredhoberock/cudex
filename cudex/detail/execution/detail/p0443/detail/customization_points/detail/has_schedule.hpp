#pragma once

#include "../../prologue.hpp"

#include <utility>
#include "../../type_traits/is_detected.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class S>
using schedule_member_function_t = decltype(std::declval<S>().schedule());

template<class S>
using has_schedule_member_function = is_detected<schedule_member_function_t, S>;


template<class S>
using schedule_free_function_t = decltype(schedule(std::declval<S>()));

template<class S>
using has_schedule_free_function = is_detected<schedule_free_function_t, S>;


} // end detail


P0443_NAMESPACE_CLOSE_BRACE

#include "../../epilogue.hpp"

