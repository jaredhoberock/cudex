#pragma once

#include "../../prologue.hpp"

#include <cstddef>
#include <utility>
#include "../../type_traits/standard_traits.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class F>
class inlined_bulk_invocable
{
  private:
    mutable F f_;
    // XXX generalize this beyond just size_t
    std::size_t shape_;

  public:
    template<class Function, class Shape>
    inlined_bulk_invocable(Function&& f, Shape&& shape)
      : f_(std::forward<F>(f)),
        shape_(std::forward<Shape>(shape))
    {}

    P0443_EXEC_CHECK_DISABLE
    P0443_ANNOTATION
    void operator()() const
    {
      for(size_t i = 0; i < shape_; ++i)
      {
        f_(i);
      }
    }
};


template<class F, class Shape>
P0443_ANNOTATION
inlined_bulk_invocable<remove_cvref_t<F>> make_inlined_bulk_invocable(F&& f, Shape&& s)
{
  return {std::forward<F>(f), std::forward<Shape>(s)};
}


} // end detail


P0443_NAMESPACE_CLOSE_BRACE

#include "../../epilogue.hpp"

