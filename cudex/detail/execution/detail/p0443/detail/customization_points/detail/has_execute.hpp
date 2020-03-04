#pragma once

#include "../../prologue.hpp"

#include <utility>
#include "../../type_traits/is_detected.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class E, class F>
using execute_member_function_t = decltype(std::declval<E>().execute(std::declval<F>()));

template<class E, class F>
using has_execute_member_function = is_detected<execute_member_function_t, E, F>;


template<class E, class F>
using execute_free_function_t = decltype(execute(std::declval<E>(), std::declval<F>()));

template<class E, class F>
using has_execute_free_function = is_detected<execute_free_function_t, E, F>;


} // end detail


P0443_NAMESPACE_CLOSE_BRACE

#include "../../epilogue.hpp"

