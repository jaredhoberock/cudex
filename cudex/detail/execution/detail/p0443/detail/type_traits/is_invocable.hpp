#pragma once

#include "../prologue.hpp"

#include <utility>
#include "is_detected.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class F, class... Args>
using invoke_t = decltype(std::declval<F>()(std::declval<Args>()...));


template<class F, class... Args>
using is_invocable = is_detected<invoke_t, F, Args...>;


} // end detail


P0443_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

