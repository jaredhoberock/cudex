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

#include "../../detail/prologue.hpp"

#include <cstdint>
#include <type_traits>
#include <utility>
#include "../../detail/functional/invoke.hpp"
#include "../../detail/type_traits/is_detected.hpp"
#include "../../detail/type_traits/is_invocable.hpp"
#include "../../detail/type_traits/remove_cvref.hpp"
#include "../../property/bulk_guarantee.hpp"
#include "../../property/detail/static_query.hpp"
#include "../execute.hpp"
#include "../executor_coordinate.hpp"
#include "../is_executor.hpp"
#include "detail/in_iteration_space.hpp"
#include "detail/next_coordinate.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class E>
using has_unsequenced_bulk_guarantee = is_detected_exact<
  bulk_guarantee_t::unsequenced_t,
  static_query_t,
  bulk_guarantee_t,
  remove_cvref_t<E>
>;


template<class Function, class Coord>
struct invoke_with_coord
{
  mutable Function f;
  Coord coord;

  CUDEX_ANNOTATION
  void operator()() const
  {
    detail::invoke(f,coord);
  }
};


CUDEX_EXEC_CHECK_DISABLE
template<class Executor, class Function,
         CUDEX_REQUIRES(is_executor<Executor>::value),
         CUDEX_REQUIRES(has_unsequenced_bulk_guarantee<Executor>::value),
         CUDEX_REQUIRES(is_invocable<remove_cvref_t<Function>,executor_coordinate_t<Executor>>::value),
         CUDEX_REQUIRES(std::is_copy_constructible<remove_cvref_t<Function>>::value)
        >
CUDEX_ANNOTATION
void default_bulk_execute(const Executor& ex, Function&& f, executor_coordinate_t<Executor> shape)
{
  using coord_type = executor_coordinate_t<Executor>;

  for(coord_type coord = coord_type{}; detail::in_iteration_space(coord, shape); detail::next_coordinate(coord, shape))
  {
    CUDEX_NAMESPACE::execute(ex, invoke_with_coord<remove_cvref_t<Function>,coord_type>{f, coord});
  }
}


template<class Function, class Coord>
struct invoke_reference_with_coord
{
  Function& f;
  Coord coord;

  CUDEX_ANNOTATION
  void operator()() const
  {
    detail::invoke(f,coord);
  }
};


CUDEX_EXEC_CHECK_DISABLE
template<class Executor, class Function,
         CUDEX_REQUIRES(is_executor<Executor>::value),
         CUDEX_REQUIRES(has_unsequenced_bulk_guarantee<Executor>::value),
         CUDEX_REQUIRES(is_invocable<remove_cvref_t<Function>&,executor_coordinate_t<Executor>>::value),
         CUDEX_REQUIRES(!std::is_copy_constructible<remove_cvref_t<Function>>::value)
        >
CUDEX_ANNOTATION
void default_bulk_execute(const Executor& ex, Function&& f, executor_coordinate_t<Executor> shape)
{
  using coord_type = executor_coordinate_t<Executor>;

  for(coord_type coord = coord_type{}; detail::in_iteration_space(coord, shape); detail::next_coordinate(coord, shape))
  {
    CUDEX_NAMESPACE::execute(ex, invoke_reference_with_coord<remove_cvref_t<Function>,coord_type>{f, coord});
  }
}


template<class E, class F, class S>
using default_bulk_execute_t = decltype(detail::default_bulk_execute(std::declval<E>(), std::declval<F>(), std::declval<S>()));


template<class E, class F, class S>
using can_default_bulk_execute = is_detected<default_bulk_execute_t, E, F, S>;


} // end detail


CUDEX_NAMESPACE_CLOSE_BRACE

#include "../../detail/epilogue.hpp"

