#pragma once

#include "../prologue.hpp"

#include "../concept_traits/is_sender_to.hpp"
#include "../sink_receiver.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class S>
using is_sender = is_sender_to<S, P0443_NAMESPACE::sink_receiver>;


} // end detail


namespace ext
{


template<class S>
using is_sender = detail::is_sender<S>;
  

} // end ext


P0443_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

