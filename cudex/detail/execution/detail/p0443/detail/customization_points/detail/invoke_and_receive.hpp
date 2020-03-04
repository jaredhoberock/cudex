#pragma once

#include "../../prologue.hpp"

#include <utility>
#include <exception>
#include "../../customization_points/detail/invoke_and_set_value.hpp"
#include "../../concept_traits/is_receiver.hpp"
#include "../../customization_points/set_error.hpp"
#include "../../type_traits/standard_traits.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class Function, class Receiver,
         P0443_REQUIRES(is_invocable<Function>::value),
         P0443_REQUIRES(is_receiver_of<Receiver, invoke_result_t<Function>>::value)
        >
P0443_ANNOTATION
void invoke_and_receive(Function&& f, Receiver&& r)
{
  try
  {
    detail::invoke_and_set_value(std::forward<Function>(f), std::forward<Receiver>(r));
    execution::set_value(std::forward<Receiver>(r));
  }
  catch(...)
  {
    execution::set_error(std::forward<Receiver>(r), std::current_exception());
  }
}


} // end detail


P0443_NAMESPACE_CLOSE_BRACE

#include "../../epilogue.hpp"

