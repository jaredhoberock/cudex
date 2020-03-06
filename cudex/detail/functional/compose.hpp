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

#include "../prologue.hpp"

#include <type_traits>
#include <utility>
#include "../type_traits/is_detected.hpp"
#include "../type_traits/is_invocable.hpp"
#include "../type_traits/invoke_result.hpp"
#include "invoke.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{
namespace compose_detail
{


// case where g returns void
template<class Invocable1, class Invocable2, class... Args,
         CUDEX_REQUIRES(is_detected_exact<void, invoke_result_t, Invocable2&&, Args&&...>::value),
         CUDEX_REQUIRES(is_invocable<Invocable1&&>::value)
        >
CUDEX_ANNOTATION
invoke_result_t<Invocable1&&> invoke_composition(Invocable1&& f, Invocable2&& g, Args&&... args)
{
  detail::invoke(std::forward<Invocable2>(g), std::forward<Args>(args)...);
  return detail::invoke(std::forward<Invocable1>(f));
}


// case where g returns non-void
template<class Invocable1, class Invocable2, class... Args>
CUDEX_ANNOTATION
auto invoke_composition(Invocable1&& f, Invocable2&& g, Args&&... args)
  -> decltype(detail::invoke(std::forward<Invocable1>(f), detail::invoke(std::forward<Invocable2>(g), std::forward<Args>(args)...)))
{
  return detail::invoke(std::forward<Invocable1>(f), detail::invoke(std::forward<Invocable2>(g), std::forward<Args>(args)...));
}


} // end compose_detail


template<class Invocable1, class Invocable2, class... Args>
using composition_invoke_result_t = decltype(compose_detail::invoke_composition(std::declval<Invocable1>(), std::declval<Invocable2>(), std::declval<Args>()...));


template<class Invocable1, class Invocable2, class... Args>
using is_composition_invocable = is_detected<composition_invoke_result_t, Invocable1, Invocable2, Args...>;


template<class Invocable1, class Invocable2>
class function_composition
{
  public:
    template<class OtherInvocable1,
             class OtherInvocable2,
             CUDEX_REQUIRES(std::is_constructible<Invocable1,OtherInvocable1&&>::value),
             CUDEX_REQUIRES(std::is_constructible<Invocable2,OtherInvocable2&&>::value)
            >
    CUDEX_ANNOTATION
    function_composition(OtherInvocable1&& f, OtherInvocable2&& g)
      : f_(std::forward<OtherInvocable1>(f)),
        g_(std::forward<OtherInvocable2>(g))
    {}

    function_composition(const function_composition&) = default;

    function_composition(function_composition&&) = default;

    template<class... Args,
             CUDEX_REQUIRES(is_composition_invocable<Invocable1&, Invocable2&, Args&&...>::value)
            >
    CUDEX_ANNOTATION
    composition_invoke_result_t<Invocable1&, Invocable2&, Args&&...>
      operator()(Args&&... args) &
    {
      return compose_detail::invoke_composition(f_, g_, std::forward<Args>(args)...);
    }

    template<class... Args,
             CUDEX_REQUIRES(is_composition_invocable<const Invocable1&, const Invocable2&, Args&&...>::value)
            >
    CUDEX_ANNOTATION
    composition_invoke_result_t<const Invocable1&, const Invocable2&, Args&&...>
      operator()(Args&&... args) const &
    {
      return compose_detail::invoke_composition(f_, g_, std::forward<Args>(args)...);
    }

    template<class... Args,
             CUDEX_REQUIRES(is_composition_invocable<Invocable1&&, Invocable2&&, Args&&...>::value)
            >
    CUDEX_ANNOTATION
    composition_invoke_result_t<Invocable1&&, Invocable2&&, Args&&...> operator()(Args&&... args) &&
    {
      return compose_detail::invoke_composition(std::move(f_), std::move(g_), std::forward<Args>(args)...);
    }

  private:
    Invocable1 f_;
    Invocable2 g_;
};


template<class Invocable1, class Invocable2>
CUDEX_ANNOTATION
function_composition<typename std::decay<Invocable1>::type, typename std::decay<Invocable2>::type> compose(Invocable1&& f, Invocable2&& g)
{
  return {std::forward<Invocable1>(f), std::forward<Invocable2>(g)};
}


} // end detail


CUDEX_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

