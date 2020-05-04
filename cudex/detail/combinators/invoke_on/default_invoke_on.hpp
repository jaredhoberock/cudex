// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "../../prologue.hpp"

#include <type_traits>
#include <utility>
#include "../../execute_operation.hpp"
#include "../../functional/closure.hpp"
#include "../../functional/compose.hpp"
#include "../../receiver_as_invocable.hpp"
#include "../../type_traits/is_invocable.hpp"
#include "../../execution.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{


// this is a sender that invokes a function on an executor and sends the result to a receiver
template<class Executor, class Invocable>
class invoke_sender : public execution::sender_base
{
  public:
    template<class OtherInvocable,
             CUDEX_REQUIRES(std::is_constructible<Invocable,OtherInvocable&&>::value)
            >
    CUDEX_ANNOTATION
    invoke_sender(const Executor& ex, OtherInvocable&& invocable)
      : ex_(ex), invocable_(std::forward<OtherInvocable>(invocable))
    {}

    invoke_sender(const invoke_sender&) = default;

    invoke_sender(invoke_sender&&) = default;

    // the type of operation returned by connect
    template<class Receiver>
    using operation = execute_operation<
      Executor,
      function_composition<
        receiver_as_invocable<Receiver>,
        Invocable
      >
    >;


    //template<class Receiver,
    //         CUDEX_REQUIRES(execution::is_receiver_of<Receiver, invoke_result_t<Invocable>>::value)
    //        >
    template<class Receiver,
             class Result = invoke_result_t<Invocable>,
             CUDEX_REQUIRES(execution::is_receiver_of<Receiver, Result>::value)
            >
    CUDEX_ANNOTATION
    operation<Receiver&&> connect(Receiver&& r) &&
    {
      auto composition = detail::compose(detail::as_invocable(std::forward<Receiver>(r)), std::move(invocable_));
      return detail::make_execute_operation(ex_, std::move(composition));
    }


    // this overload allows makes invoke_sender a "multi-shot" sender when Invocable is copyable
    template<class Receiver,
             CUDEX_REQUIRES(std::is_copy_constructible<Invocable>::value),
             CUDEX_REQUIRES(execution::is_receiver_of<Receiver, invoke_result_t<Invocable>>::value)
            >
    CUDEX_ANNOTATION
    operation<Receiver&&> connect(Receiver&& r) const &
    {
      auto composition = detail::compose(detail::as_invocable(std::forward<Receiver>(r)), invocable_);
      return detail::make_execute_operation(ex_, std::move(composition));
    }

    
    template<class OtherExecutor,
             CUDEX_REQUIRES(execution::is_executor<OtherExecutor>::value)
            >
    CUDEX_ANNOTATION
    invoke_sender<OtherExecutor, Invocable> on(const OtherExecutor& ex) &&
    {
      return {ex, std::move(invocable_)};
    }


    template<class OtherExecutor,
             CUDEX_REQUIRES(execution::is_executor<OtherExecutor>::value),
             CUDEX_REQUIRES(std::is_copy_constructible<Invocable>::value)
            >
    CUDEX_ANNOTATION
    invoke_sender<OtherExecutor, Invocable> on(const OtherExecutor& ex) const &
    {
      return {ex, invocable_};
    }

  private:
    Executor ex_;
    Invocable invocable_;
};


template<class Executor, class Invocable,
         CUDEX_REQUIRES(detail::execution::is_executor_of<Executor,Invocable>::value)
        >
CUDEX_ANNOTATION
invoke_sender<Executor, decay_t<Invocable>>
  default_invoke_on(const Executor& ex, Invocable&& f)
{
  return {ex, std::forward<Invocable>(f)};
}


template<class Executor, class Invocable,
         class Arg1, class... Args,
         CUDEX_REQUIRES(detail::execution::is_executor<Executor>::value),
         CUDEX_REQUIRES(detail::is_invocable<Invocable,Arg1,Args...>::value)
        >
CUDEX_ANNOTATION
invoke_sender<Executor, closure<decay_t<Invocable>, decay_t<Arg1>, decay_t<Args>...>>
  default_invoke_on(const Executor& ex, Invocable&& f, Arg1&& arg1, Args&&... args)
{
  return detail::default_invoke_on(ex, detail::bind(std::forward<Invocable>(f), std::forward<Arg1>(arg1), std::forward<Args>(args)...));
}


} // end namespace detail


CUDEX_NAMESPACE_CLOSE_BRACE


#include "../../epilogue.hpp"

