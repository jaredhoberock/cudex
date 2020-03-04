#pragma once

#include "../../prologue.hpp"

#include <utility>
#include <type_traits>
#include "../../concept_traits/is_receiver_of.hpp"
#include "../../customization_points/detail/invoke.hpp"
#include "../../customization_points/set_value.hpp"
#include "../../type_traits/is_invocable.hpp"
#include "../../type_traits/standard_traits.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class Function, class Receiver,
         P0443_REQUIRES(is_invocable<Function>::value),
         P0443_REQUIRES(is_receiver_of<Receiver, invoke_result_t<Function>>::value),
         P0443_REQUIRES(std::is_void<invoke_result_t<Function>>::value)
        >
P0443_ANNOTATION
void invoke_and_set_value(Function&& f, Receiver&& r)
{
  detail::invoke(f);
  P0443_NAMESPACE::set_value(std::forward<Receiver>(r));
}


template<class Function, class Receiver,
         P0443_REQUIRES(is_invocable<Function>::value),
         P0443_REQUIRES(is_receiver_of<Receiver, invoke_result_t<Function>>::value),
         P0443_REQUIRES(!std::is_void<invoke_result_t<Function>>::value)
        >
P0443_ANNOTATION
void invoke_and_set_value(Function&& f, Receiver&& r)
{
  P0443_NAMESPACE::set_value(std::forward<Receiver>(r), detail::invoke(f));
}


} // end detail


P0443_NAMESPACE_CLOSE_BRACE

#include "../../epilogue.hpp"

