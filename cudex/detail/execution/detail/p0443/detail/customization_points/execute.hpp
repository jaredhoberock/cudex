#pragma once

#include "../prologue.hpp"

#include <utility>
#include "../customization_points/detail/has_execute.hpp"
#include "../customization_points/detail/has_submit.hpp"
#include "../customization_points/detail/invocable_as_receiver.hpp"
#include "../customization_points/detail/static_const.hpp"
#include "../type_traits/is_detected.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


// this is the type of execute
struct execute_customization_point
{
  P0443_EXEC_CHECK_DISABLE
  template<class E, class F,
           P0443_REQUIRES(has_execute_member_function<E&&,F&&>::value)
          >
  P0443_ANNOTATION
  constexpr auto operator()(E&& e, F&& f) const ->
    decltype(std::forward<E>(e).execute(std::forward<F>(f)))
  {
    return std::forward<E>(e).execute(std::forward<F>(f));
  }

  P0443_EXEC_CHECK_DISABLE
  template<class E, class F,
           P0443_REQUIRES(!has_execute_member_function<E&&,F&&>::value and
                          has_execute_free_function<E&&,F&&>::value)
          >
  P0443_ANNOTATION
  constexpr auto operator()(E&& e, F&& f) const ->
    decltype(execute(std::forward<E>(e), std::forward<F>(f)))
  {
    return execute(std::forward<E>(e), std::forward<F>(f));
  }

  P0443_EXEC_CHECK_DISABLE
  template<class E, class F,
           P0443_REQUIRES(!has_execute_member_function<E&&,F&&>::value and
                          !has_execute_free_function<E&&,F&&>::value and
                          has_submit_member_function<E&&,invocable_as_receiver<F&&>>::value)
          >
  P0443_ANNOTATION
  constexpr auto operator()(E&& e, F&& f) const ->
    decltype(std::forward<E>(e).submit(detail::as_receiver(std::forward<F>(f))))
  {
    // note that this overload does not use execution::submit to avoid circular dependency
    return std::forward<E>(e).submit(detail::as_receiver(std::forward<F>(f)));
  }

  P0443_EXEC_CHECK_DISABLE
  template<class E, class F,
           P0443_REQUIRES(!has_execute_member_function<E&&,F&&>::value and
                          !has_execute_free_function<E&&,F&&>::value and
                          !has_submit_member_function<E&&,invocable_as_receiver<F&&>>::value and
                          has_submit_free_function<E&&,invocable_as_receiver<F&&>>::value)
          >
  P0443_ANNOTATION
  constexpr auto operator()(E&& e, F&& f) const ->
    decltype(submit(std::forward<E>(e), detail::as_receiver(std::forward<F>(f))))
  {
    // note that this overload does not use execution::submit to avoid circular dependency
    return submit(std::forward<E>(e), detail::as_receiver(std::forward<F>(f)));
  }
};


} // end detail


namespace
{


// define the execute customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& execute = detail::static_const<detail::execute_customization_point>::value;
#else
const __device__ detail::execute_customization_point execute;
#endif


} // end anonymous namespace


P0443_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

