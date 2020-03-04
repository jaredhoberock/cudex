#pragma once

#include "../prologue.hpp"

#include <utility>
#include "../customization_points/detail/static_const.hpp"
#include "../type_traits/is_detected.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class R, class E>
using set_error_member_t = decltype(std::declval<R>().set_error(std::declval<E>()));


template<class R, class E>
using has_set_error_member = is_detected<set_error_member_t, R, E>;


template<class R, class E>
using set_error_free_function_t = decltype(set_error(std::declval<R>(), std::declval<E>()));


template<class R, class E>
using has_set_error_free_function = is_detected<set_error_free_function_t, R, E>;


// this is the type of set_error
struct set_error_customization_point
{
  P0443_EXEC_CHECK_DISABLE
  template<class R, class E,
           P0443_REQUIRES(has_set_error_member<R&&,E&&>::value)
          >
  P0443_ANNOTATION
  constexpr auto operator()(R&& r, E&& e) const ->
    decltype(std::forward<R>(r).set_error(std::forward<E>(e)))
  {
    return std::forward<R>(r).set_error(std::forward<E>(e));
  }

  P0443_EXEC_CHECK_DISABLE
  template<class R, class E,
           P0443_REQUIRES(!has_set_error_member<R&&,E&&>::value and
                          has_set_error_free_function<R&&,E&&>::value)
          >
  P0443_ANNOTATION
  constexpr auto operator()(R&& r, E&& e) const ->
    decltype(set_error(std::forward<R>(r), std::forward<E>(e)))
  {
    return set_error(std::forward<R>(r), std::forward<E>(e));
  }
};


} // end detail



namespace
{


// define the set_error customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& set_error = detail::static_const<detail::set_error_customization_point>::value;
#else
const __device__ detail::set_error_customization_point set_error;
#endif


} // end anonymous namespace


P0443_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

