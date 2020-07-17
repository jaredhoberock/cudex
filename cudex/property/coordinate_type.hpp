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

#include <cstdint>
#include "../detail/type_traits/is_detected.hpp"
#include "../executor/is_executor.hpp"
#include "detail/basic_executor_property.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


template<class T = void>
struct coordinate_type_property :
// XXX TODO implement requirability
//  detail::basic_executor_property<coordinate_type_property<T>, true, false>
  detail::basic_executor_property<coordinate_type_property<T>, false, false>
{
  using type = T;


  CUDEX_ANNOTATION
  friend constexpr bool operator==(const coordinate_type_property&, const coordinate_type_property&)
  {
    return true;
  }

  CUDEX_ANNOTATION
  friend constexpr bool operator!=(const coordinate_type_property&, const coordinate_type_property&)
  {
    return false;
  }


  //template<class Executor,
  //         CUDEX_REQUIRES(is_executor<Executor>::value)
  //        >
  //CUDEX_ANNOTATION
  //friend constexpr coordinate_adaptor<Executor,T> require(Executor&& ex, coordinate_type_property)
  //{
  //  return {std::forward<Executor>(ex)};
  //}
};


// this type exists to enable query(ex, coordinate_type<>)
template<>
class coordinate_type_property<void> :
  public detail::basic_executor_property<coordinate_type_property<void>, false, false>
{
  public:
    // XXX C++11 doesn't have variable templates so define this as a constexpr
    // function in the meantime
    template<class Executor>
    CUDEX_ANNOTATION
    constexpr static coordinate_type_property<typename Executor::coordinate_type> static_query()
    {
      return {};
    }


  private:
    template<class T>
    using nested_coordinate_type_t = typename T::coordinate_type;


  public:
    template<class Executor,
             CUDEX_REQUIRES(is_executor<Executor>::value),
             CUDEX_REQUIRES(detail::is_detected<nested_coordinate_type_t, Executor>::value)
            >
    CUDEX_ANNOTATION
    friend constexpr coordinate_type_property<nested_coordinate_type_t<Executor>>
      query(Executor&&, coordinate_type_property)
    {
      return {};
    }


    // by default, an executor's coordinate_type is std::size_t
    template<class Executor,
             CUDEX_REQUIRES(is_executor<Executor>::value),
             CUDEX_REQUIRES(!detail::is_detected<nested_coordinate_type_t, Executor>::value)
            >
    CUDEX_ANNOTATION
    friend constexpr coordinate_type_property<std::size_t> query(Executor&&, coordinate_type_property)
    {
      return {};
    }
};


#ifndef __CUDA_ARCH__
template<class T = void>
static constexpr coordinate_type_property<T> coordinate_type{};
#else
template<class T = void>
const __device__ coordinate_type_property<T> coordinate_type;
#endif


CUDEX_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

