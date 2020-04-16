#pragma once

#include "../prologue.hpp"

#include <type_traits>
#include "../sender_traits.hpp"
#include "../type_traits/standard_traits.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class T>
using is_typed_sender = detail::conjunction<is_sender<T>, has_sender_types<remove_cvref_t<T>>>;


} // end detail


namespace ext
{


template<class T>
using is_typed_sender = detail::is_typed_sender<T>;


} // end ext


P0443_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

