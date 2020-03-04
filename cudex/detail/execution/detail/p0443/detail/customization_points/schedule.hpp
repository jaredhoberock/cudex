#pragma once

#include "../prologue.hpp"

#include <utility>
#include "../concept_traits/is_sender.hpp"
#include "../customization_points/detail/decay_copy.hpp"
#include "../customization_points/detail/has_schedule.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


// this is the type of schedule
struct schedule_customization_point
{
  P0443_EXEC_CHECK_DISABLE
  template<class S,
           P0443_REQUIRES(has_schedule_member_function<S&&>::value)
          >
  P0443_ANNOTATION
  constexpr auto operator()(S&& s) const ->
    decltype(std::forward<S>(s).schedule())
  {
    return std::forward<S>(s).schedule();
  }

  P0443_EXEC_CHECK_DISABLE
  template<class S,
           P0443_REQUIRES(!has_schedule_member_function<S&&>::value and
                          has_schedule_free_function<S&&>::value)
          >
  P0443_ANNOTATION
  constexpr auto operator()(S&& s) const ->
    decltype(schedule(std::forward<S>(s)))
  {
    return schedule(std::forward<S>(s));
  }

  P0443_EXEC_CHECK_DISABLE
  template<class S,
           P0443_REQUIRES(!has_schedule_member_function<S&&>::value and
                          !has_schedule_free_function<S&&>::value and
                          is_sender<S&&>::value)
          >
  P0443_ANNOTATION
  constexpr auto operator()(S&& s) const
    -> decltype(detail::decay_copy(std::forward<S>(s)))
  {
    return detail::decay_copy(std::forward<S>(s));
  }
};
  

} // end detail


namespace
{


// define the schedule customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& schedule = detail::static_const<detail::schedule_customization_point>::value;
#else
const __device__ detail::schedule_customization_point schedule;
#endif


} // end anonymous namespace


P0443_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

