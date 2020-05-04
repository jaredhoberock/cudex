#pragma once

#include "../detail/prologue.hpp"

#include <utility>
#include "../detail/static_const.hpp"
#include "../detail/type_traits/is_detected.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class E, class F>
using execute_member_function_t = decltype(std::declval<E>().execute(std::declval<F>()));

template<class E, class F>
using has_execute_member_function = is_detected<execute_member_function_t, E, F>;


template<class E, class F>
using execute_free_function_t = decltype(execute(std::declval<E>(), std::declval<F>()));

template<class E, class F>
using has_execute_free_function = is_detected<execute_free_function_t, E, F>;


// this is the type of execute
struct dispatch_execute
{
  CUDEX_EXEC_CHECK_DISABLE
  template<class E, class F,
           CUDEX_REQUIRES(has_execute_member_function<E&&,F&&>::value)
          >
  CUDEX_ANNOTATION
  constexpr auto operator()(E&& e, F&& f) const ->
    decltype(std::forward<E>(e).execute(std::forward<F>(f)))
  {
    return std::forward<E>(e).execute(std::forward<F>(f));
  }

  CUDEX_EXEC_CHECK_DISABLE
  template<class E, class F,
           CUDEX_REQUIRES(!has_execute_member_function<E&&,F&&>::value and
                          has_execute_free_function<E&&,F&&>::value)
          >
  CUDEX_ANNOTATION
  constexpr auto operator()(E&& e, F&& f) const ->
    decltype(execute(std::forward<E>(e), std::forward<F>(f)))
  {
    return execute(std::forward<E>(e), std::forward<F>(f));
  }
};


} // end detail


namespace
{


// define the execute customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& execute = detail::static_const<detail::dispatch_execute>::value;
#else
const __device__ detail::dispatch_execute execute;
#endif


} // end anonymous namespace


CUDEX_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

