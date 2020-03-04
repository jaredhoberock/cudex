#pragma once

#include "../prologue.hpp"

#include <utility>
#include "../type_traits/is_detected.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class L, class R>
using operator_equal_t = decltype(std::declval<L>() == std::declval<R>());


template<class T>
using is_equality_comparable = is_detected<operator_equal_t, T, T>;


} // end detail


P0443_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

