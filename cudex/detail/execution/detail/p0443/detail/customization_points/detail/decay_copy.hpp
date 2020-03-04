#pragma once

#include "../../prologue.hpp"

#include <utility>
#include <type_traits>


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


P0443_EXEC_CHECK_DISABLE
template<class T>
P0443_ANNOTATION
constexpr typename std::decay<T&&>::type decay_copy(T&& arg)
{
  return std::forward<T>(arg);
}


} // end detail


P0443_NAMESPACE_CLOSE_BRACE

#include "../../epilogue.hpp"

