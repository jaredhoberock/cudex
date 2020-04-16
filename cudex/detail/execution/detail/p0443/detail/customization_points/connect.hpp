#pragma once

#include "../prologue.hpp"

#include <utility>
#include "../customization_points/detail/has_connect.hpp"
#include "../customization_points/detail/static_const.hpp"
#include "../type_traits/is_detected.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


// this is the type of connect
struct connect_customization_point
{
  P0443_EXEC_CHECK_DISABLE
  template<class S, class R,
           P0443_REQUIRES(has_connect_member_function<S&&,R&&>::value)
          >
  P0443_ANNOTATION
  constexpr auto operator()(S&& s, R&& r) const ->
    decltype(std::forward<S>(s).connect(std::forward<R>(r)))
  {
    return std::forward<S>(s).connect(std::forward<R>(r));
  }

  P0443_EXEC_CHECK_DISABLE
  template<class S, class R,
           P0443_REQUIRES(!has_connect_member_function<S&&,R&&>::value and
                          has_connect_free_function<S&&,R&&>::value)
          >
  P0443_ANNOTATION
  constexpr auto operator()(S&& s, R&& r) const ->
    decltype(connect(std::forward<S>(s), std::forward<R>(r)))
  {
    return connect(std::forward<S>(s), std::forward<R>(r));
  }

  // XXX consider what connect's default implementation, if any, should be
};


} // end detail


namespace
{


// define the connect customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& connect = detail::static_const<detail::connect_customization_point>::value;
#else
const __device__ detail::connect_customization_point connect;
#endif


} // end anonymous namespace


namespace detail
{


template<class S, class R>
using connect_t = decltype(P0443_NAMESPACE::connect(std::declval<S>(), std::declval<R>()));


} // end detail


P0443_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

