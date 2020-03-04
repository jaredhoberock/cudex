#pragma once

#include "../../prologue.hpp"

#include <exception>
#include <utility>
#include "../../concept_traits/is_receiver.hpp"
#include "../../type_traits/is_invocable.hpp"
#include "../../type_traits/standard_traits.hpp"
#include "executor_operation.hpp"
#include "inlined_bulk_invocable.hpp"
#include "received_invocable.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class E, class F>
class bulk_sender
{
  private:
    E ex_;
    inlined_bulk_invocable<F> func_;

    template<class R>
    using executable_t = received_invocable<inlined_bulk_invocable<F>, R>;

  public:
    P0443_EXEC_CHECK_DISABLE
    template<class Executor, class Function, class Shape>
    P0443_ANNOTATION
    bulk_sender(Executor&& ex, Function&& f, Shape&& shape)
      : ex_(std::forward<Executor>(ex)),
        func_(std::forward<F>(f), std::forward<Shape>(shape))
    {}

    // connects this sender to a receiver to produce an executor_operation
    template<class R,
             P0443_REQUIRES(is_receiver_of<R&&>::value)
            >
    P0443_ANNOTATION
    executor_operation<E, executable_t<remove_cvref_t<R>>> connect(R&& r)
    {
      // connect the completion of the bulk invocation to the receiver
      executable_t<remove_cvref_t<R>> executable{std::move(func_), std::forward<R>(r)};

      // package up the executor and the executable
      return {ex_, std::move(executable)};
    }
};


template<class E, class F, class S,
         P0443_REQUIRES(is_executor<E&&>::value),
         P0443_REQUIRES(is_invocable<F&&,S&&>::value)
        >
P0443_ANNOTATION
bulk_sender<remove_cvref_t<E>, remove_cvref_t<F>> make_bulk_sender(E&& e, F&& f, S&& s)
{
  return {std::forward<E>(e), std::forward<F>(f), std::forward<S>(s)};
}


// this trait deduces the type of make_bulk_sender's result
template<class E, class F, class S>
using make_bulk_sender_t = decltype(make_bulk_sender(std::declval<E>(), std::declval<F>(), std::declval<S>()));


} // end detail


P0443_NAMESPACE_CLOSE_BRACE

#include "../../epilogue.hpp"

