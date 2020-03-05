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

#include <cstdint>


CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{


template<std::size_t... I> struct index_sequence {};

template<std::size_t Start, typename Indices, std::size_t End>
struct make_index_sequence_impl;

template<std::size_t Start, std::size_t... Indices, std::size_t End>
struct make_index_sequence_impl<
  Start,
  index_sequence<Indices...>, 
  End
>
{
  using type = typename make_index_sequence_impl<
    Start + 1,
    index_sequence<Indices..., Start>,
    End
  >::type;
};

template<std::size_t End, std::size_t... Indices>
struct make_index_sequence_impl<End, index_sequence<Indices...>, End>
{
  using type = index_sequence<Indices...>;
};

template<std::size_t N>
using make_index_sequence = typename make_index_sequence_impl<0, index_sequence<>, N>::type;


} // end detail


CUDEX_NAMESPACE_CLOSE_BRACE


#include "../epilogue.hpp"

