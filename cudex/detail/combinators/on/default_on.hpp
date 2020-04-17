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
#include "../../execution.hpp"
#include "../../functional/closure.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class Sender, class Executor>
class on_sender : public execution::sender_base
{
  public:
    CUDEX_EXEC_CHECK_DISABLE
    template<class OtherSender,
             CUDEX_REQUIRES(std::is_constructible<Sender,OtherSender&&>::value)
            >
    CUDEX_ANNOTATION
    on_sender(OtherSender&& sender, const Executor& executor)
      : sender_(std::forward<OtherSender>(sender)),
        executor_(executor)
    {}

    on_sender(const on_sender&) = default;

    on_sender(on_sender&&) = default;

    CUDEX_ANNOTATION
    const Executor& get_executor() const
    {
      return executor_;
    }

    // the type of operation returned by on_sender::connect
    template<class Receiver>
    class operation
    {
      public:
        CUDEX_EXEC_CHECK_DISABLE
        CUDEX_ANNOTATION
        operation(const Executor& executor, Sender&& sender, Receiver&& receiver)
          : executor_(executor),
            sender_(std::move(sender)),
            receiver_(std::move(receiver))
        {}

        CUDEX_ANNOTATION
        void start() &&
        {
          // create a function that will submit the sender to the receiver and then execute that function on our executor
          execution::execute(executor_, detail::bind(execution::submit, std::move(sender_), std::move(receiver_)));
        }

      private:
        Executor executor_;
        Sender sender_;
        Receiver receiver_;
    };

    template<class Receiver,
             CUDEX_REQUIRES(execution::is_sender_to<Sender&&,Receiver&&>::value)
            >
    CUDEX_ANNOTATION
    operation<decay_t<Receiver>> connect(Receiver&& r) &&
    {
      return {executor_, std::move(sender_), std::move(r)};
    }

    template<class OtherExecutor,
             CUDEX_REQUIRES(execution::is_executor<OtherExecutor>::value)
            >
    CUDEX_ANNOTATION
    on_sender<Sender, OtherExecutor> on(const OtherExecutor& executor) &&
    {
      return {std::move(sender_), executor};
    }

    template<class OtherExecutor,
             CUDEX_REQUIRES(execution::is_executor<OtherExecutor>::value),
             CUDEX_REQUIRES(std::is_copy_constructible<Sender>::value)
            >
    CUDEX_ANNOTATION
    on_sender<Sender, OtherExecutor> on(const OtherExecutor& executor) const &
    {
      return {sender_, executor};
    }

  private:
    Sender sender_;
    Executor executor_;
};


template<class Sender, class Executor,
         CUDEX_REQUIRES(detail::execution::is_sender<Sender>::value),
         CUDEX_REQUIRES(detail::execution::is_executor<Executor>::value)
        >
CUDEX_ANNOTATION
on_sender<decay_t<Sender>, Executor> default_on(Sender&& s, const Executor& ex)
{
  return {std::forward<Sender>(s), ex};
}


} // end namespace detail


CUDEX_NAMESPACE_CLOSE_BRACE


#include "../../epilogue.hpp"

