#pragma once

#include "../../prologue.hpp"

#include <utility>
#include <type_traits>


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class T>
P0443_ANNOTATION
typename std::conditional<  
    !std::is_nothrow_move_constructible<T>::value && std::is_copy_constructible<T>::value,
    const T&,
    T&&
>::type move_if_noexcept(T& x) noexcept
{
  return std::move(x);
}


} // end detail


P0443_NAMESPACE_CLOSE_BRACE

#include "../../epilogue.hpp"

