#pragma once

#include "../../prologue.hpp"

#include <utility>
#include "../../type_traits/is_detected.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class S, class R>
using submit_member_function_t = decltype(std::declval<S>().submit(std::declval<R>()));

template<class S, class R>
using has_submit_member_function = is_detected<submit_member_function_t, S, R>;


template<class S, class R>
using submit_free_function_t = decltype(submit(std::declval<S>(), std::declval<R>()));

template<class S, class R>
using has_submit_free_function = is_detected<submit_free_function_t, S, R>;


} // end detail


P0443_NAMESPACE_CLOSE_BRACE

#include "../../epilogue.hpp"

