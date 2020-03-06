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

#include <utility>
#include "execution.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class Executor, class Invocable>
class execute_operation
{
  public:
    template<class OtherInvocable,
             CUDEX_REQUIRES(std::is_constructible<Invocable,OtherInvocable&&>::value)
            >
    CUDEX_ANNOTATION
    execute_operation(const Executor& ex, OtherInvocable&& f)
      : ex_(ex), f_(std::forward<OtherInvocable>(f))
    {}

    execute_operation(const execute_operation&) = default;

    CUDEX_ANNOTATION
    execute_operation(execute_operation&& other)
      : ex_(std::move(other.ex_)), f_(std::move(other.f_))
    {}

    template<CUDEX_REQUIRES(execution::is_executor_of<Executor,Invocable&&>::value)>
    CUDEX_ANNOTATION
    void start() &&
    {
      execution::execute(ex_, std::move(f_));
    }

    template<CUDEX_REQUIRES(execution::is_executor_of<Executor,const Invocable&>::value)>
    CUDEX_ANNOTATION
    void start() const &
    {
      execution::execute(ex_, f_);
    }

    template<CUDEX_REQUIRES(execution::is_executor_of<Executor,Invocable&>::value)>
    CUDEX_ANNOTATION
    void start() &
    {
      execution::execute(ex_, f_);
    }

  private:
    Executor ex_;
    Invocable f_;
};


template<class Executor, class Invocable,
         CUDEX_REQUIRES(execution::is_executor<Executor>::value)
        >
CUDEX_ANNOTATION
execute_operation<Executor, typename std::decay<Invocable>::type> make_execute_operation(const Executor& ex, Invocable&& f)
{
  return {ex, std::forward<Invocable>(f)};
}


} // end detail


CUDEX_NAMESPACE_CLOSE_BRACE


#include "epilogue.hpp"

