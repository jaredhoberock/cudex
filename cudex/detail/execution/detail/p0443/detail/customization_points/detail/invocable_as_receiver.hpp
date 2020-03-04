#pragma once

#include "../../prologue.hpp"

#include <exception>
#include "../../customization_points/detail/invoke.hpp"
#include "../../customization_points/detail/move_if_noexcept.hpp"
#include "../../type_traits/standard_traits.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class F>
class invocable_as_receiver
{
  private:
    using invocable_type = remove_cvref_t<F>;
    invocable_type f_;

  public:
    P0443_EXEC_CHECK_DISABLE
    P0443_ANNOTATION
    explicit invocable_as_receiver(invocable_type&& f)
      : f_(detail::move_if_noexcept(f))
    {}

    P0443_EXEC_CHECK_DISABLE
    P0443_ANNOTATION
    explicit invocable_as_receiver(const invocable_type& f)
      : f_(f)
    {}

    invocable_as_receiver(invocable_as_receiver&& other) = default;

    P0443_EXEC_CHECK_DISABLE
    P0443_ANNOTATION
    void set_value() {
      detail::invoke(f_);
    }

    void set_error(std::exception_ptr) {
      std::terminate();
    }

    P0443_EXEC_CHECK_DISABLE
    P0443_ANNOTATION
    void set_done() noexcept {}
};


template<class F>
P0443_ANNOTATION
invocable_as_receiver<F> as_receiver(F&& f)
{
  return invocable_as_receiver<F>{std::forward<F>(f)};
}


} // end detail


P0443_NAMESPACE_CLOSE_BRACE

#include "../../epilogue.hpp"

