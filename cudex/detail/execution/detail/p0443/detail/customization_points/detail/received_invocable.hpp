#pragma once

#include "../../prologue.hpp"

#include <exception>
#include <utility>
#include "../set_error.hpp"
#include "../set_done.hpp"
#include "invoke_and_receive.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


// XXX this thing is sort of like a packaged_task to which you provide a promise (aka a receiver)
template<class F, class R>
class received_invocable
{
  private:
    F function_;

    // XXX better to use std::optional when available
    R receiver_;
    bool valid_;

  public:
    P0443_EXEC_CHECK_DISABLE
    template<class Function, class Receiver>
    P0443_ANNOTATION
    received_invocable(Function&& function, Receiver&& receiver)
      try : function_(std::forward<Function>(function)),
            receiver_(std::forward<Receiver>(receiver)),
            valid_(true)
      {}
      catch(...)
      {
        execution::set_error(receiver, std::current_exception());
      }

    P0443_EXEC_CHECK_DISABLE
    P0443_ANNOTATION
    received_invocable(received_invocable&& other)
      try : function_(std::move(other.function_)),
            receiver_(std::move(other.receiver_)),
            valid_(other.valid_)
      {
        other.valid_ = false;
      }
      catch(...)
      {
        execution::set_error(other.receiver_, std::current_exception());
      }

    P0443_ANNOTATION
    ~received_invocable()
    {
      if(valid_)
      {
        execution::set_done(receiver_);
      }
    }

    P0443_ANNOTATION
    void operator()()
    {
      detail::invoke_and_receive(function_, receiver_);
      valid_ = false;
    }
};


template<class Function, class Receiver,
         P0443_REQUIRES(is_invocable<Function>::value and
                        is_receiver_of<Receiver, invoke_result_t<Function>>::value)
        >
P0443_ANNOTATION
received_invocable<remove_cvref_t<Function>, remove_cvref_t<Receiver>> make_received_invocable(Function&& function, Receiver&& receiver)
{
  return {std::forward<Function>(function), std::forward<Receiver>(receiver)};
}


} // end detail


P0443_NAMESPACE_CLOSE_BRACE

#include "../../epilogue.hpp"

