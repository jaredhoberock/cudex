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

#include "prologue.hpp"

#include <type_traits>
#include <utility>
#include "dispatch_then.hpp"
#include "execution.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{


// this template wraps another sender and introduces member functions for convenient chaining
template<class Sender>
class chaining_sender
{
  private:
    Sender sender_;

  public:
    template<class OtherSender,
             CUDEX_REQUIRES(std::is_constructible<Sender,OtherSender&&>::value)
            >
    CUDEX_ANNOTATION
    chaining_sender(OtherSender&& sender)
      : sender_(std::forward<OtherSender>(sender))
    {}

    chaining_sender(const chaining_sender&) = default;

    chaining_sender(chaining_sender&&) = default;


    template<class Receiver>
    CUDEX_ANNOTATION
    auto connect(Receiver&& receiver) &&
      -> decltype(execution::connect(std::move(sender_), std::forward<Receiver>(receiver)))
    {
      return execution::connect(std::move(sender_), std::forward<Receiver>(receiver));
    }


    template<class Receiver,
             CUDEX_REQUIRES(execution::is_sender_to<Sender&&,Receiver&&>::value)
            >
    CUDEX_ANNOTATION
    void submit(Receiver&& receiver) &&
    {
      execution::submit(std::move(sender_), std::forward<Receiver>(receiver));
    }


    template<class Function,
             CUDEX_REQUIRES(can_dispatch_then<Sender&&,Function&&>::value)
            >
    CUDEX_ANNOTATION
    chaining_sender<dispatch_then_t<Sender&&,Function&&>>
      then(Function&& continuation) &&
    {
      return {detail::dispatch_then(std::move(sender_), std::forward<Function>(continuation))};
    }
};


} // end detail

CUDEX_NAMESPACE_CLOSE_BRACE

#include "epilogue.hpp"

