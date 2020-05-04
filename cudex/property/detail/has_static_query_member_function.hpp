#pragma once

#include "../../detail/prologue.hpp"

#include <utility>
#include "../../detail/type_traits/is_detected.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class T, class P>
using static_query_member_function_t = decltype(T::query(std::declval<P>()));

template<class T, class P>
using has_static_query_member_function = is_detected<static_query_member_function_t, T, P>;


} // end detail


CUDEX_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

