#pragma once

#include "../../prologue.hpp"

#include <utility>
#include "../../type_traits/is_detected.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class T>
using start_member_function_t = decltype(std::declval<T>().start());

template<class T>
using has_start_member_function = is_detected<start_member_function_t, T>;


template<class T>
using start_free_function_t = decltype(start(std::declval<T>()));

template<class T>
using has_start_free_function = is_detected<start_free_function_t, T>;


} // end detail


P0443_NAMESPACE_CLOSE_BRACE

#include "../../epilogue.hpp"

