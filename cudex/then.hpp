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

#include "detail/prologue.hpp"

#include <utility>
#include "detail/execution.hpp"
#include "detail/functional/invoke.hpp"
#include "detail/type_traits/decay.hpp"
#include "detail/type_traits/invoke_result.hpp"
#include "detail/type_traits/is_nothrow_invocable.hpp"
#include "detail/type_traits/is_nothrow_receiver_of.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class Receiver, class Function>
class then_receiver
{
  public:
    CUDEX_ANNOTATION
    then_receiver(Receiver receiver, Function f)
      : receiver_(std::move(receiver)), f_(std::move(f))
    {}

    template<class... Args, class Result = invoke_result_t<Function, Args...>,
             CUDEX_REQUIRES(execution::is_receiver_of<Receiver, Result>::value)
            >
    CUDEX_ANNOTATION
    void set_value(Args&&... args) &&
        noexcept(is_nothrow_invocable<Function, Args...>::value and is_nothrow_receiver_of<Receiver, Result>::value)
    {
      execution::set_value(std::move(receiver_), detail::invoke(std::move(f_), std::forward<Args>(args)...));
    }

//    template<class... Argss, class Ret = invoke_result_t<F, As...>>
//        requires receiver_of<R> && std::is_void_v<Ret>
//    void set_value(Argss&&... args) &&
//        noexcept(std::is_nothrow_invocable_v<F, As...> &&
//            is_nothrow_receiver_of_v<R>)
//    {
//      std::invoke(std::move(f_), std::forward<Args>(args)...);
//      execution::set_value(std::move(receiver_));
//    }

    template<class Error,
             CUDEX_REQUIRES(execution::is_receiver<Receiver, Error>::value)
            >
    CUDEX_ANNOTATION
    void set_error(Error&& error) && noexcept
    {
      execution::set_error(std::move(receiver_), std::forward<Error>(error));
    }

    CUDEX_ANNOTATION
    void set_done() && noexcept
    {
      execution::set_done(std::move(receiver_));
    }

  private:
    Receiver receiver_;
    Function f_;
};


template<class Receiver, class Function>
CUDEX_ANNOTATION
then_receiver<decay_t<Receiver>, decay_t<Function>> make_then_receiver(Receiver&& receiver, Function&& continuation)
{
  return {std::forward<Receiver>(receiver), std::forward<Function>(continuation)};
}


template<class Sender, class Function>
class then_sender
{
  private:
    Sender predecessor_;
    Function continuation_;

  public:
    template<class OtherSender, class OtherFunction,
             CUDEX_REQUIRES(std::is_constructible<Sender,OtherSender&&>::value),
             CUDEX_REQUIRES(std::is_constructible<Function,OtherFunction&&>::value)
            >
    CUDEX_ANNOTATION
    then_sender(OtherSender&& predecessor, OtherFunction&& continuation)
      : predecessor_(std::forward<OtherSender>(predecessor)), continuation_(std::forward<OtherFunction>(continuation))
    {}

    then_sender(const then_sender&) = default;

    then_sender(then_sender&&) = default;

    template<class Receiver,
             CUDEX_REQUIRES(execution::is_sender_to<Sender, then_receiver<Receiver, Function>>::value)
            >
    CUDEX_ANNOTATION
    auto connect(Receiver&& r) &&
      -> decltype(execution::connect(std::move(predecessor_), detail::make_then_receiver(std::move(r), std::move(continuation_))))
    {
      return execution::connect(std::move(predecessor_), detail::make_then_receiver(std::move(r), std::move(continuation_)));
    }
};


} // end detail


template<class Sender, class Function>
CUDEX_ANNOTATION
detail::then_sender<detail::decay_t<Sender>, detail::decay_t<Function>>
  then(Sender&& predecessor, Function&& continuation)
{
  return {std::forward<Sender>(predecessor), std::forward<Function>(continuation)};
}


CUDEX_NAMESPACE_CLOSE_BRACE


#include "detail/epilogue.hpp"

