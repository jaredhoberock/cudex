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

#if __has_include(<any>)
#include <any>
#endif

#include "detail/basic_executor_property.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


template<class ProtoAllocator>
class allocator_t : 
  detail::basic_executor_property<
    allocator_t<ProtoAllocator>,
    true,
    true
#if __cpp_lib_any
    , std::any
#endif
  >
{
  public:
    CUDEX_ANNOTATION
    constexpr explicit allocator_t(const ProtoAllocator& alloc) : alloc_(alloc) {}

    CUDEX_ANNOTATION
    constexpr ProtoAllocator value() const
    {
      return alloc_;
    }

  private:
    ProtoAllocator alloc_;
};


template<>
struct allocator_t<void> :
  detail::basic_executor_property<
    allocator_t<void>,
    true,
    true
  >
{
  template<class ProtoAllocator>
  CUDEX_ANNOTATION
  constexpr allocator_t<ProtoAllocator> operator()(const ProtoAllocator& alloc) const
  {
    return allocator_t<ProtoAllocator>{alloc};
  }
};


#ifndef __CUDA_ARCH__
static constexpr allocator_t<void> allocator{};
#else
const __device__ allocator_t<void> allocator;
#endif


CUDEX_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

