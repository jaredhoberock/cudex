#pragma once

#include "../prologue.hpp"

#include <utility>
#include "../customization_points/detail/static_const.hpp"
#include "../type_traits/is_detected.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class R>
using set_done_member_function_t = decltype(std::declval<R>().set_done());

template<class R>
using has_set_done_member_function = is_detected<set_done_member_function_t, R>;


template<class R>
using set_done_free_function_t = decltype(set_done(std::declval<R>()));

template<class R>
using has_set_done_free_function = is_detected<set_done_free_function_t, R>;



// this is the type of set_done
struct set_done_customization_point
{
  P0443_EXEC_CHECK_DISABLE
  template<class R,
           P0443_REQUIRES(has_set_done_member_function<R&&>::value)
          >
  P0443_ANNOTATION
  constexpr auto operator()(R&& r) const ->
    decltype(std::forward<R>(r).set_done())
  {
    return std::forward<R>(r).set_done();
  }

  P0443_EXEC_CHECK_DISABLE
  template<class R,
           P0443_REQUIRES(!has_set_done_member_function<R&&>::value and
                          has_set_done_free_function<R&&>::value)
           >
  P0443_ANNOTATION
  constexpr auto operator()(R&& r) const ->
    decltype(set_done(std::forward<R>(r)))
  {
    return set_done(std::forward<R>(r));
  }
};


} // end detail



namespace
{


// define the set_done customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& set_done = detail::static_const<detail::set_done_customization_point>::value;
#else
const __device__ detail::set_done_customization_point set_done;
#endif


} // end anonymous namespace


P0443_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

