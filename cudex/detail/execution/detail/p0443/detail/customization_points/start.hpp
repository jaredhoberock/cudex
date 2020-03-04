#pragma once

#include "../prologue.hpp"

#include <utility>
#include "../type_traits/is_detected.hpp"
#include "detail/static_const.hpp"
#include "detail/has_start.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


// this is the type of start
struct start_customization_point
{
  P0443_EXEC_CHECK_DISABLE
  template<class O, 
           P0443_REQUIRES(has_start_member_function<O&&>::value)
          >
  P0443_ANNOTATION
  constexpr auto operator()(O&& o) const ->
    decltype(std::forward<O>(o).start())
  {
    return std::forward<O>(o).start();
  }

  P0443_EXEC_CHECK_DISABLE
  template<class O,
           P0443_REQUIRES(!has_start_member_function<O&&>::value and
                          has_start_free_function<O&&>::value)
          >
  P0443_ANNOTATION
  constexpr auto operator()(O&& o) const ->
    decltype(start(std::forward<O>(o)))
  {
    return start(std::forward<O>(o));
  }
};


} // end detail


namespace
{


// define the start customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& start = detail::static_const<detail::start_customization_point>::value;
#else
const __device__ detail::start_customization_point start;
#endif


} // end anonymous namespace


P0443_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"


