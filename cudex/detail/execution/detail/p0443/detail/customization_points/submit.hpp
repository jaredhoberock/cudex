#pragma once

#include "../prologue.hpp"

#include <utility>
#include "../type_traits/is_detected.hpp"
#include "detail/connect_and_start.hpp"
#include "detail/has_execute.hpp"
#include "detail/has_submit.hpp"
#include "detail/receiver_as_invocable.hpp"
#include "detail/static_const.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class S, class R>
using can_connect_and_start = is_detected<connect_and_start_t, S, R>;


// this is the type of submit
struct submit_customization_point
{
  P0443_EXEC_CHECK_DISABLE
  template<class S, class R,
           P0443_REQUIRES(has_submit_member_function<S&&,R&&>::value)
          >
  P0443_ANNOTATION
  constexpr auto operator()(S&& s, R&& r) const ->
    decltype(std::forward<S>(s).submit(std::forward<R>(r)))
  {
    return std::forward<S>(s).submit(std::forward<R>(r));
  }

  P0443_EXEC_CHECK_DISABLE
  template<class S, class R,
           P0443_REQUIRES(!has_submit_member_function<S&&,R&&>::value and
                          has_submit_free_function<S&&,R&&>::value)
          >
  P0443_ANNOTATION
  constexpr auto operator()(S&& s, R&& r) const ->
    decltype(submit(std::forward<S>(s), std::forward<R>(r)))
  {
    return submit(std::forward<S>(s), std::forward<R>(r));
  }

  P0443_EXEC_CHECK_DISABLE
  template<class S, class R,
           P0443_REQUIRES(!has_submit_member_function<S&&,R&&>::value and
                          !has_submit_free_function<S&&,R&&>::value and
                          has_execute_member_function<S&&,receiver_as_invocable<R&&>>::value)
          >
  P0443_ANNOTATION
  constexpr auto operator()(S&& s, R&& r) const ->
    decltype(std::forward<S>(s).execute(detail::as_invocable(std::forward<R>(r))))
  {
    // note that this overload does not use execution::execute to avoid circular dependency
    return std::forward<S>(s).execute(detail::as_invocable(std::forward<R>(r)));
  }

  P0443_EXEC_CHECK_DISABLE
  template<class S, class R,
           P0443_REQUIRES(!has_submit_member_function<S&&,R&&>::value and
                          !has_submit_free_function<S&&,R&&>::value and
                          !has_execute_member_function<S&&,receiver_as_invocable<R&&>>::value and
                          has_execute_free_function<S&&,receiver_as_invocable<R&&>>::value)
          >
  P0443_ANNOTATION
  constexpr auto operator()(S&& s, R&& r) const ->
    decltype(submit(std::forward<S>(s), detail::as_invocable(std::forward<R>(r))))
  {
    // note that this overload does not use execution::execute to avoid circular dependency
    return execute(std::forward<S>(s), detail::as_invocable(std::forward<R>(r)));
  }

  P0443_EXEC_CHECK_DISABLE
  template<class S, class R,
           P0443_REQUIRES(!has_submit_member_function<S&&,R&&>::value and
                          !has_submit_free_function<S&&,R&&>::value and
                          !has_execute_member_function<S&&,receiver_as_invocable<R&&>>::value and
                          !has_execute_free_function<S&&,receiver_as_invocable<R&&>>::value and
                          can_connect_and_start<S&&,R&&>::value)
          >
  P0443_ANNOTATION
  void operator()(S&& s, R&& r) const
  {
    detail::connect_and_start(std::forward<S>(s), std::forward<R>(r));
  }
};


} // end detail


namespace
{


// define the submit customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& submit = detail::static_const<detail::submit_customization_point>::value;
#else
const __device__ detail::submit_customization_point submit;
#endif


} // end anonymous namespace


P0443_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

