#pragma once

#include "../prologue.hpp"

#include "is_executor_of.hpp"
#include "../invocable_archetype.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class E>
using is_executor = is_executor_of<E, invocable_archetype>;


} // end detail


namespace ext
{


template<class E>
using is_executor = detail::is_executor<E>;


} // end ext


P0443_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

