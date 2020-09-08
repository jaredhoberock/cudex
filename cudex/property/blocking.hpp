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

#include "../executor/is_executor.hpp"
#include "detail/basic_executor_property.hpp"
#include "detail/has_static_query_member_function.hpp"

CUDEX_NAMESPACE_OPEN_BRACE


class blocking_t : 
  public detail::basic_executor_property<blocking_t, false, false, blocking_t>
{
  private:
    int which_;

    CUDEX_ANNOTATION
    friend constexpr bool operator==(const blocking_t& a, const blocking_t& b)
    {
      return a.which_ == b.which_;
    }

    CUDEX_ANNOTATION
    friend constexpr bool operator!=(const blocking_t& a, const blocking_t& b)
    {
      return !(a == b);
    }

  public:
    CUDEX_ANNOTATION
    constexpr blocking_t()
      : which_{0}
    {}


    struct possibly_t :
      detail::basic_executor_property<possibly_t, true, true, blocking_t>
    {
      CUDEX_ANNOTATION
      static constexpr possibly_t value()
      {
        return possibly_t{};
      }
    };

    static constexpr possibly_t possibly{};

    CUDEX_ANNOTATION
    constexpr blocking_t(const possibly_t&)
      : which_{1}
    {}


    struct always_t :
      detail::basic_executor_property<always_t, true, false, blocking_t> // the inconsistent true, false used here is intentional
    {
      CUDEX_ANNOTATION
      static constexpr always_t value()
      {
        return always_t{};
      }
    };

    static constexpr always_t always{};

    CUDEX_ANNOTATION
    constexpr blocking_t(const always_t&)
      : which_{2}
    {}


    struct never_t :
      detail::basic_executor_property<never_t, true, true, blocking_t>
    {
      CUDEX_ANNOTATION
      static constexpr never_t value()
      {
        return never_t{};
      }
    };

    static constexpr never_t never{};

    CUDEX_ANNOTATION
    constexpr blocking_t(const never_t&)
      : which_{3}
    {}


    // By default, executors are possibly blocking if blocking_t cannot
    // be statically-queried through a member
    template<class Executor,
             CUDEX_REQUIRES(
               !detail::has_static_query_member_function<Executor, blocking_t>::value
             )>
    CUDEX_ANNOTATION
    constexpr static possibly_t static_query()
    {
      return possibly_t{};
    }


    template<class Executor,
             CUDEX_REQUIRES(
               detail::has_static_query_member_function<Executor, blocking_t>::value
             )>
    CUDEX_ANNOTATION
    static constexpr auto static_query()
      -> decltype(Executor::query(std::declval<blocking_t>()))
    {
      return Executor::query(blocking_t{});
    }
};


static constexpr blocking_t blocking{};


// older C++ requires definitions for static constexpr members
#if __cplusplus < 201703L
constexpr blocking_t::possibly_t blocking_t::possibly;
constexpr blocking_t::always_t blocking_t::always;
constexpr blocking_t::never_t blocking_t::never;
#endif


CUDEX_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

