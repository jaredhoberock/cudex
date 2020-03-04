#pragma once

#include "../prologue.hpp"

P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class T, class P>
class is_applicable_property
{
  private:
    template<class T_, class P_,
             bool result = P_::template is_applicable_property<T_>()
            >
    constexpr static bool test(int)
    {
      return P_::template is_applicable_property_v<T_>;
    }

    template<class...>
    constexpr static bool test(...)
    {
      return false;
    }

  public:
    using value_type = bool;
    constexpr static value_type value = test<T,P>(0);
    using type = std::integral_constant<value_type, value>;
};


} // end detail


P0443_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

