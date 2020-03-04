#pragma once

#include "../../prologue.hpp"

#include "../../concept_traits/is_executor.hpp"
#include "../../static_query.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class Derived, bool requireable, bool preferable, class polymorphic_query_result_type_ = void>
struct basic_executor_property
{
  static constexpr bool is_requirable = requireable;
  static constexpr bool is_preferable = preferable;

  using polymorphic_query_result_type = polymorphic_query_result_type_;

  // XXX c++11 doesn't have variable templates so define this as a constexpr
  // function in the meantime
  template<class T>
  static constexpr bool is_applicable_property()
  {
    return detail::is_executor<T>::value;
  }

  // XXX C++11 doesn't have variable templates so define this as a constexpr
  // function in the meantime
  template<class Executor>
  P0443_ANNOTATION
  static constexpr auto static_query() ->
    decltype(detail::static_query<Executor,Derived>())
  {
    return detail::static_query<Executor,Derived>();
  }


#if __cplusplus >= 201402L
  template<class T>
  static constexpr bool is_applicable_property_v = is_applicable_property<T>();

  template<class Executor>
  static constexpr decltype(auto) static_query_v = static_query<Executor>();
#endif
};


} // end detail


P0443_NAMESPACE_CLOSE_BRACE


#include "../../epilogue.hpp"

