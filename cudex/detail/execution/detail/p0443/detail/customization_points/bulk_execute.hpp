#pragma once

#include "../prologue.hpp"

#include <utility>
#include "../customization_points/detail/bulk_sender.hpp"
#include "../customization_points/detail/has_bulk_execute.hpp"
#include "../customization_points/detail/static_const.hpp"
#include "../properties/bulk_guarantee.hpp"
#include "../static_query.hpp"
#include "../type_traits/standard_traits.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class E, class F, class S>
using can_make_bulk_unsequenced_sender = conjunction<
  is_detected<
    make_bulk_sender_t,E,F,S
  >
  , is_detected_exact<
    bulk_guarantee_t::unsequenced_t,
    static_query_t,
    bulk_guarantee_t,
    typename std::decay<E>::type
  >
>;


// this is the type of bulk_execute
struct bulk_execute_customization_point
{
  // try a member function
  P0443_EXEC_CHECK_DISABLE
  template<class E, class F, class S,
           P0443_REQUIRES(has_bulk_execute_member_function<E&&,F&&,S&&>::value)
          >
  P0443_ANNOTATION
  constexpr auto operator()(E&& e, F&& f, S&& s) const ->
    decltype(std::forward<E>(e).bulk_execute(std::forward<F>(f), std::forward<S>(s)))
  {
    return std::forward<E>(e).bulk_execute(std::forward<F>(f), std::forward<S>(s));
  }

  // try a free function
  P0443_EXEC_CHECK_DISABLE
  template<class E, class F, class S,
           P0443_REQUIRES(!has_bulk_execute_member_function<E&&,F&&,S&&>::value and
                          has_bulk_execute_free_function<E&&,F&&,S&&>::value)
          >
  P0443_ANNOTATION
  constexpr auto operator()(E&& e, F&& f, S&& s) const ->
    decltype(bulk_execute(std::forward<E>(e), std::forward<F>(f), std::forward<S>(s)))
  {
    return bulk_execute(std::forward<E>(e), std::forward<F>(f), std::forward<S>(s));
  }

  // default path
  P0443_EXEC_CHECK_DISABLE
  template<class E, class F, class S,
           P0443_REQUIRES(!has_bulk_execute_member_function<E&&,F&&,S&&>::value and
                          !has_bulk_execute_free_function<E&&,F&&,S&&>::value and
                          can_make_bulk_unsequenced_sender<E&&,F&&,S&&>::value)
          >
  P0443_ANNOTATION
  constexpr auto operator()(E&& e, F&& f, S&& s) const ->
    decltype(detail::make_bulk_sender(std::forward<E>(e), std::forward<F>(f), std::forward<S>(s)))
  {
    return detail::make_bulk_sender(std::forward<E>(e), std::forward<F>(f), std::forward<S>(s));
  }
};


} // end detail


namespace
{


// define the bulk_execute customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& bulk_execute = detail::static_const<detail::bulk_execute_customization_point>::value;
#else
const __device__ detail::bulk_execute_customization_point bulk_execute;
#endif


} // end anonymous namespace


P0443_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

