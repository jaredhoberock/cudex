#pragma once

#include "../prologue.hpp"

#include <type_traits>


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class T>
using is_nothrow_move_or_copy_constructible = std::integral_constant<
  bool,
  std::is_nothrow_move_constructible<T>::value or
  std::is_copy_constructible<T>::value
>;


} // end detail


P0443_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

