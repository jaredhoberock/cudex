#pragma once

#include "prologue.hpp"

#include "type_traits/is_detected.hpp"
#include "type_traits/standard_traits.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


P0443_EXEC_CHECK_DISABLE
template<class T, class P,
         class R = decltype(P::template static_query<T>())>
P0443_ANNOTATION
constexpr R static_query(const P&)
{
  return P::template static_query<T>();
}


template<class P, class T>
using static_query_t = decltype(static_query<remove_cvref_t<T>>(std::declval<P>()));

template<class P, class T>
using has_static_query = is_detected<static_query_t, P, T>;


} // end detail


P0443_NAMESPACE_CLOSE_BRACE

#include "epilogue.hpp"

