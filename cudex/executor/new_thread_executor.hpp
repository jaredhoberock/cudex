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

#include "../detail/prologue.hpp"

#include <thread>
#include <utility>
#include "../detail/functional/invoke.hpp"
#include "../detail/type_traits.hpp"
#include "../property/blocking.hpp"
#include "is_executor.hpp"

CUDEX_NAMESPACE_OPEN_BRACE


struct new_thread_executor
{
  template<class Function,
           CUDEX_REQUIRES(detail::is_invocable<Function&&>::value)
          >
  void execute(Function&& f) const noexcept
  {
    std::thread(std::forward<Function>(f)).detach();
  }

  CUDEX_ANNOTATION
  constexpr bool operator==(const new_thread_executor&) const noexcept
  {
    return true;
  }

  CUDEX_ANNOTATION
  constexpr bool operator!=(const new_thread_executor&) const noexcept
  {
    return false;
  }

  CUDEX_ANNOTATION
  constexpr static blocking_t query(blocking_t)
  {
    return blocking.never;
  }
};


static_assert(is_executor<new_thread_executor>::value, "Error.");


CUDEX_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

