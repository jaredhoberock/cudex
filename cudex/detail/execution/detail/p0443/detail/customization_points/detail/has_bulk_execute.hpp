#pragma once

#include "../../prologue.hpp"

#include <utility>
#include "../../type_traits/is_detected.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class E, class F, class S>
using bulk_execute_member_function_t = decltype(std::declval<E>().bulk_execute(std::declval<F>(), std::declval<S>()));

template<class E, class F, class S>
using has_bulk_execute_member_function = is_detected<bulk_execute_member_function_t, E, F, S>;


template<class E, class F, class S>
using bulk_execute_free_function_t = decltype(bulk_execute(std::declval<E>(), std::declval<F>(), std::declval<S>()));

template<class E, class F, class S>
using has_bulk_execute_free_function = is_detected<bulk_execute_free_function_t, E, F, S>;


} // end detail


P0443_NAMESPACE_CLOSE_BRACE

#include "../../epilogue.hpp"

