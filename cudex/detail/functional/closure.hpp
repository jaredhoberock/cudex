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
#include "../tuple.hpp"
#include "../type_traits.hpp"
#include "apply.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class Invocable, class... Args>
using is_bindable = disjunction<
  is_applicable<Invocable&, tuple<Args...>&>,
  is_applicable<const Invocable&, const tuple<Args...>&>,
  is_applicable<Invocable&&, tuple<Args...>&&>
>;


template<class Invocable, class... Args>
class closure 
{
  private:
    using tuple_type = tuple<Args...>;

  public:
    template<class OtherInvocable, class... OtherArgs,
             CUDEX_REQUIRES(std::is_constructible<Invocable, OtherInvocable&&>::value),
             CUDEX_REQUIRES(std::is_constructible<tuple_type, OtherArgs&&...>::value)
            >
    CUDEX_ANNOTATION
    closure(OtherInvocable&& f, OtherArgs&&... args)
      : f_(std::forward<OtherInvocable>(f)),
        args_(std::forward<OtherArgs>(args)...)
    {}


    closure(const closure&) = default;

    
    // indirect use of Invocable via a defaulted parameter enables SFINAE in CUDEX_REQUIRES
    template<class I = Invocable, CUDEX_REQUIRES(is_applicable<I&,tuple_type&>::value)>
    CUDEX_ANNOTATION
    apply_result_t<I&,tuple_type&> operator()() &
    {
      return detail::apply(f_, args_);
    }


    // indirect use of Invocable via a defaulted parameter enables SFINAE in CUDEX_REQUIRES
    template<class I = Invocable, CUDEX_REQUIRES(is_applicable<const I&,const tuple_type&>::value)>
    CUDEX_ANNOTATION
    apply_result_t<const I&,const tuple_type&> operator()() const &
    {
      return detail::apply(f_, args_);
    }


    // indirect use of Invocable via a defaulted parameter enables SFINAE in CUDEX_REQUIRES
    template<class I = Invocable, CUDEX_REQUIRES(is_applicable<I&&,tuple_type&&>::value)>
    CUDEX_ANNOTATION
    apply_result_t<I&&,tuple_type&&> operator()() &&
    {
      return detail::apply(std::move(f_), std::move(args_));
    }


  private:
    Invocable f_;
    tuple_type args_;
};


template<class Invocable, class... Args,
         CUDEX_REQUIRES(is_bindable<decay_t<Invocable>, decay_t<Args>...>::value)
        >
CUDEX_ANNOTATION
closure<decay_t<Invocable>, decay_t<Args>...>
  bind(Invocable&& f, Args&&... args)
{
  return {std::forward<Invocable>(f), std::forward<Args>(args)...};
}


} // end detail


CUDEX_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

