#pragma once

#include "../prologue.hpp"

#include <utility>
#include "detail/static_const.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class R, class... Args>
using set_value_member_function_t = decltype(std::declval<R>().set_value(std::declval<Args>()...));

template<class R, class... Args>
using has_set_value_member_function = is_detected<set_value_member_function_t, R, Args...>;


template<class R, class... Args>
using set_value_free_function_t = decltype(set_value(std::declval<R>(), std::declval<Args>()...));

template<class R, class... Args>
using has_set_value_free_function = is_detected<set_value_free_function_t, R, Args...>;


// this is the type of set_value
struct set_value_customization_point
{
  P0443_EXEC_CHECK_DISABLE
  template<class R, class... Args,
           P0443_REQUIRES(has_set_value_member_function<R&&,Args&&...>::value)
          >
  P0443_ANNOTATION
  constexpr auto operator()(R&& r, Args&&... args) const ->
    decltype(std::forward<R>(r).set_value(std::forward<Args>(args)...))
  {
    return std::forward<R>(r).set_value(std::forward<Args>(args)...);
  }

  P0443_EXEC_CHECK_DISABLE
  template<class R, class... Args,
           P0443_REQUIRES(!has_set_value_member_function<R&&,Args&&...>::value and
                          has_set_value_free_function<R&&,Args&&...>::value)
           >
  P0443_ANNOTATION
  constexpr auto operator()(R&& r, Args&&... args) const ->
    decltype(set_value(std::forward<R>(r), std::forward<Args>(args)...))
  {
    return set_value(std::forward<R>(r), std::forward<Args>(args)...);
  }
};


} // end detail



namespace
{


// define the set_value customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& set_value = detail::static_const<detail::set_value_customization_point>::value;
#else
const __device__ detail::set_value_customization_point set_value;
#endif


} // end anonymous namespace


P0443_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

