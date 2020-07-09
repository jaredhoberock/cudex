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

#include "../detail/type_traits/conjunction.hpp"
#include "../detail/type_traits/disjunction.hpp"
#include "../property/stream.hpp"
#include "is_executor.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class T>
using stream_member_function_t = decltype(std::declval<T>().stream());


template<class T>
using stream_query_member_function_t = decltype(std::declval<T>().query(std::declval<stream_property>()));


template<class T>
using stream_query_free_function_t = decltype(query(std::declval<T>(), std::declval<stream_property>()));


template<class T>
using has_stream_property = disjunction<
  is_detected<stream_member_function_t, T>,
  is_detected<stream_query_member_function_t, T>,
  is_detected<stream_query_free_function_t, T>
>;


} // end detail


template<class T>
using is_device_executor = detail::conjunction<
  is_executor<T>,
  // XXX ideally, we'd use can_query here, but cudex has not defined that trait
  //     nor the query customization point
  //     instead, just use something ad hoc
  //can_query<T,stream_property>
  detail::has_stream_property<T>
>;


CUDEX_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

