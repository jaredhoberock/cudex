#pragma once

#include "../../prologue.hpp"

#include <utility>
#include "../../type_traits/is_detected.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class S, class R>
using connect_member_function_t = decltype(std::declval<S>().connect(std::declval<R>()));

template<class S, class R>
using has_connect_member_function = is_detected<connect_member_function_t, S, R>;


template<class S, class R>
using connect_free_function_t = decltype(connect(std::declval<S>(), std::declval<R>()));

template<class S, class R>
using has_connect_free_function = is_detected<connect_free_function_t, S, R>;


} // end detail


P0443_NAMESPACE_CLOSE_BRACE

#include "../../epilogue.hpp"

