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

#include <type_traits>
#include <utility>
#include "detail/combinators/on/dispatch_on.hpp"
#include "detail/combinators/then/dispatch_then.hpp"
#include "detail/execution.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


// this template wraps another sender and introduces convenient chaining via member functions
template<class Sender>
class chaining_sender
{
  private:
    Sender sender_;

  public:
    CUDEX_EXEC_CHECK_DISABLE
    template<class OtherSender,
             CUDEX_REQUIRES(detail::execution::is_sender<OtherSender&&>::value),
             CUDEX_REQUIRES(std::is_constructible<Sender,OtherSender&&>::value)
            >
    CUDEX_ANNOTATION
    chaining_sender(OtherSender&& sender)
      : sender_{std::forward<OtherSender>(sender)}
    {}

    CUDEX_EXEC_CHECK_DISABLE
    CUDEX_ANNOTATION
    chaining_sender(const chaining_sender& other)
      : sender_{other.sender_}
    {}

    CUDEX_EXEC_CHECK_DISABLE
    CUDEX_ANNOTATION
    chaining_sender(chaining_sender&& other)
      : sender_{std::move(other.sender_)}
    {}


    CUDEX_EXEC_CHECK_DISABLE
    CUDEX_ANNOTATION
    ~chaining_sender() {}


    template<class Receiver,
             CUDEX_REQUIRES(detail::execution::is_sender_to<Sender&&,Receiver&&>::value)
            >
    CUDEX_ANNOTATION
    auto connect(Receiver&& receiver) &&
      -> decltype(detail::execution::connect(std::move(sender_), std::forward<Receiver>(receiver)))
    {
      return detail::execution::connect(std::move(sender_), std::forward<Receiver>(receiver));
    }


    template<class Receiver,
             CUDEX_REQUIRES(detail::execution::is_sender_to<Sender&&,Receiver&&>::value)
            >
    CUDEX_ANNOTATION
    void submit(Receiver&& receiver) &&
    {
      detail::execution::submit(std::move(sender_), std::forward<Receiver>(receiver));
    }


    template<class Function,
             CUDEX_REQUIRES(detail::can_dispatch_then<Sender&&,Function&&>::value)
            >
    CUDEX_ANNOTATION
    chaining_sender<detail::dispatch_then_t<Sender&&,Function&&>>
      then(Function&& continuation) &&
    {
      return {detail::dispatch_then(std::move(sender_), std::forward<Function>(continuation))};
    }


    template<class Executor,
             CUDEX_REQUIRES(detail::can_dispatch_on<Sender&&,const Executor&>::value)
            >
    CUDEX_ANNOTATION
    chaining_sender<detail::dispatch_on_t<Sender&&,const Executor&>>
      on(const Executor& ex) &&
    {
      return {detail::dispatch_on(std::move(sender_), ex)};
    }
};


namespace detail
{
namespace execution
{


// specialize sender_traits
template<class Sender>
struct sender_traits<chaining_sender<Sender>> : sender_traits<Sender> {};


} // end execution
} // end detail


// this utility allows clients (such as the sender combinator CPOs) to ensure that the senders they return
// aren't multiply-wrapped chaining_senders
// i.e., chaining_sender<chaining_sender<...>> is unhelpful, so let's avoid creating those
template<class Sender,
         CUDEX_REQUIRES(detail::execution::is_sender<Sender&&>::value),
         CUDEX_REQUIRES(std::is_rvalue_reference<Sender&&>::value)
        >
CUDEX_ANNOTATION
chaining_sender<detail::decay_t<Sender>> ensure_chaining_sender(Sender&& sender)
{
  return {std::move(sender)};
}

template<class Sender>
CUDEX_ANNOTATION
chaining_sender<Sender> ensure_chaining_sender(chaining_sender<Sender>&& sender)
{
  return std::move(sender);
}


template<class Sender>
using ensure_chaining_sender_t = decltype(ensure_chaining_sender(std::declval<Sender>()));


// make multiply-wrapped chaining_senders illegal for now 
// XXX eliminate this ASAP, there might be some use case for multiple-wrapping
template<class Sender>
class chaining_sender<chaining_sender<Sender>>;


CUDEX_NAMESPACE_CLOSE_BRACE

#include "detail/epilogue.hpp"

