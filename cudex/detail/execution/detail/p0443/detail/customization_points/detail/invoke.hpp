#pragma once

#include "../../prologue.hpp"

#include <utility>
#include <type_traits>


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


P0443_EXEC_CHECK_DISABLE
template<class F, class... Args>
P0443_ANNOTATION
constexpr auto invoke(F&& f, Args&&... args)
  -> decltype(std::forward<F>(f)(std::forward<Args>(args)...))
{
  // XXX this needs to be generalized to pointers to member functions etc.
  return std::forward<F>(f)(std::forward<Args>(args)...);
}


} // end detail


P0443_NAMESPACE_CLOSE_BRACE

#include "../../epilogue.hpp"

