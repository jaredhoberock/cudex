#pragma once

#include "../prologue.hpp"

#include "../concept_traits/is_executor.hpp"
#include "../static_query.hpp"
#include "detail/has_query.hpp"
#include "detail/basic_executor_property.hpp"


P0443_NAMESPACE_OPEN_BRACE


struct bulk_guarantee_t :
  detail::basic_executor_property<bulk_guarantee_t, false, false, bulk_guarantee_t>
{
  P0443_ANNOTATION
  friend constexpr bool operator==(const bulk_guarantee_t& a, const bulk_guarantee_t& b)
  {
    return a.which_ == b.which_;
  }

  P0443_ANNOTATION
  friend constexpr bool operator!=(const bulk_guarantee_t& a, const bulk_guarantee_t& b)
  {
    return !(a == b);
  }

  P0443_ANNOTATION
  constexpr bulk_guarantee_t()
    : which_{0}
  {}

  struct sequenced_t :
    detail::basic_executor_property<sequenced_t, true, true>
  {
    P0443_ANNOTATION
    static constexpr sequenced_t value()
    {
      return sequenced_t{};
    }
  };

  static constexpr sequenced_t sequenced{};


  P0443_ANNOTATION
  constexpr bulk_guarantee_t(const sequenced_t&)
    : which_{1}
  {}

  struct parallel_t :
    detail::basic_executor_property<parallel_t, true, true>
  {
    P0443_ANNOTATION
    static constexpr parallel_t value()
    {
      return parallel_t{};
    }
  };

  static constexpr parallel_t parallel{};

  P0443_ANNOTATION
  constexpr bulk_guarantee_t(const parallel_t&)
    : which_{2}
  {}


  struct unsequenced_t :
    detail::basic_executor_property<unsequenced_t, true, true>
  {
    P0443_ANNOTATION
    static constexpr unsequenced_t value()
    {
      return unsequenced_t{};
    }
  };

  static constexpr unsequenced_t unsequenced{};

  P0443_ANNOTATION
  constexpr bulk_guarantee_t(const unsequenced_t&)
    : which_{3}
  {}


  // By default, executors are unsequenced if bulk_guarantee_t cannot
  // be statically-queried through a member
  template<class Executor,
           P0443_REQUIRES(
             !detail::has_static_query_member_function<Executor, bulk_guarantee_t>::value
           )>
  P0443_ANNOTATION
  static constexpr unsequenced_t static_query()
  {
    return unsequenced_t{};
  }

  private:
    int which_;
};


static constexpr bulk_guarantee_t bulk_guarantee{};


P0443_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

