#pragma once

#include "../../prologue.hpp"

#include <utility>
#include "../../type_traits/is_detected.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class T, class P>
using static_query_member_function_t = decltype(T::template query(std::declval<P>()));

template<class T, class P>
using has_static_query_member_function = is_detected<static_query_member_function_t, T, P>;


} // end detail


P0443_NAMESPACE_CLOSE_BRACE

#include "../../epilogue.hpp"

