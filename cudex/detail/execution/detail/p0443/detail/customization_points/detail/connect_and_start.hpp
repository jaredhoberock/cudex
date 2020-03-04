#pragma once

#include "../../prologue.hpp"

#include <type_traits>
#include <utility>
#include "../../type_traits/is_detected.hpp"
#include "../connect.hpp"
#include "../start.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class Sender, class Receiver>
using connect_t = decltype(P0443_NAMESPACE::connect(std::declval<Sender>(), std::declval<Receiver>()));


template<class Sender, class Receiver>
using can_connect = is_detected<connect_t, Sender, Receiver>;


template<class Operation>
using start_t = decltype(P0443_NAMESPACE::start(std::declval<Operation>()));

template<class Operation>
using can_start = is_detected<start_t, Operation>;


P0443_EXEC_CHECK_DISABLE
template<class Sender, class Receiver,
         P0443_REQUIRES(can_connect<Sender&&,Receiver&&>::value),
         P0443_REQUIRES(can_start<connect_t<Sender&&,Receiver&&>>::value)
        >
P0443_ANNOTATION
void connect_and_start(Sender&& s, Receiver&& r)
{
  P0443_NAMESPACE::start(P0443_NAMESPACE::connect(std::forward<Sender>(s), std::forward<Receiver>(r)));
}


template<class Sender, class Receiver>
using connect_and_start_t = decltype(detail::connect_and_start(std::declval<Sender>(), std::declval<Receiver>()));


} // end detail


P0443_NAMESPACE_CLOSE_BRACE

#include "../../epilogue.hpp"

