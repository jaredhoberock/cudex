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

#include <exception>
#include <utility>
#include "execution.hpp"
#include "type_traits.hpp"
#include "utility/move_if_noexcept.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class R>
class receiver_as_invocable
{
  private:
    using receiver_type = remove_cvref_t<R>;

    static_assert(execution::is_receiver<receiver_type>::value, "Error.h");

    // XXX better to replace these members with a __host__ __device__ optional<receiver_type> type when available
    //     once we do that, we can eliminate the CUDEX_EXEC_CHECK_DISABLE below
    receiver_type r_;
    bool valid_;


  public:
    CUDEX_EXEC_CHECK_DISABLE
    CUDEX_ANNOTATION
    explicit receiver_as_invocable(receiver_type&& r)
#if CUDEX_HAS_EXCEPTIONS
      try : r_(detail::move_if_noexcept(r)), valid_(true) {}
      catch(...)
      {
        execution::set_error(detail::move_if_noexcept(r), std::current_exception());
      }
#else
      : r_(detail::move_if_noexcept(r)), valid_(true) {}
#endif


    CUDEX_EXEC_CHECK_DISABLE
    CUDEX_ANNOTATION
    explicit receiver_as_invocable(const receiver_type& r)
#if CUDEX_HAS_EXCEPTIONS
      try : r_(r), valid_(true) {}
      catch(...)
      {
        execution::set_error(r, std::current_exception());
      }
#else
      : r_(r), valid_(true) {}
#endif


#if CUDEX_HAS_EXCEPTIONS
    CUDEX_EXEC_CHECK_DISABLE
    CUDEX_ANNOTATION
    receiver_as_invocable(receiver_as_invocable&& other)
      try : r_(detail::move_if_noexcept(other.r_)), valid_(other.valid_)
      {
        other.valid_ = false;
      }
      catch(...)
      {
        execution::set_error(detail::move_if_noexcept(other.r_), std::current_exception());
        other.valid_ = false;
      }
#else
    CUDEX_EXEC_CHECK_DISABLE
    CUDEX_ANNOTATION
    receiver_as_invocable(receiver_as_invocable&& other)
      : r_(detail::move_if_noexcept(other.r_)), valid_(other.valid_)
    {
      other.valid_ = false;
    }
#endif


    CUDEX_EXEC_CHECK_DISABLE
    CUDEX_ANNOTATION
    receiver_as_invocable(const receiver_as_invocable& other) noexcept
      : r_(other.r_), valid_(other.valid_)
    {}


    CUDEX_EXEC_CHECK_DISABLE
    CUDEX_ANNOTATION
    ~receiver_as_invocable()
    {
      if(valid_)
      {
        execution::set_done(std::move(r_));
      }
    }


#if CUDEX_HAS_EXCEPTIONS
    template<class... Args,
             CUDEX_REQUIRES(execution::is_receiver_of<receiver_type, Args&&...>::value)
            >
    CUDEX_ANNOTATION
    void operator()(Args&&... args)
    {
      try
      {
        execution::set_value(std::move(r_), std::forward<Args>(args)...);
      }
      catch(...)
      {
        execution::set_error(std::move(r_), std::current_exception());
      }

      valid_ = false;
    }
#else
    template<class... Args,
             CUDEX_REQUIRES(execution::is_receiver_of<receiver_type, Args&&...>::value)
            >
    CUDEX_ANNOTATION
    void operator()(Args&&... args)
    {
      execution::set_value(std::move(r_), std::forward<Args>(args)...);
      valid_ = false;
    }
#endif
};


template<class R,
         CUDEX_REQUIRES(detail::execution::is_receiver<R>::value)
        >
CUDEX_ANNOTATION
receiver_as_invocable<R&&> as_invocable(R&& r)
{
  return receiver_as_invocable<R&&>{std::forward<R>(r)};
}


} // end detail


CUDEX_NAMESPACE_CLOSE_BRACE

#include "epilogue.hpp"

