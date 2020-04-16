#pragma once

#include "../prologue.hpp"

#include <type_traits>
#include "../customization_points/connect.hpp"
#include "../type_traits/is_detected.hpp"
#include "../type_traits/standard_traits.hpp"
#include "is_sender.hpp"
#include "is_receiver.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class S, class R>
using is_sender_to = conjunction<
  is_sender<S>,
  is_receiver<R>,
  is_detected<connect_t, S, R>
>;


} // end detail


namespace ext
{


template<class S, class R>
using is_sender_to = detail::is_sender_to<S,R>;
  

} // end ext


P0443_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

