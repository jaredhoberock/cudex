#pragma once

#include "../../prologue.hpp"

#include "../../concept_traits/is_executor.hpp"
#include "../../type_traits/is_invocable.hpp"
#include "../../type_traits/standard_traits.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class E, class F>
class executor_operation
{
  private:
    E executor_;
    F function_;

  public:
    P0443_EXEC_CHECK_DISABLE
    template<class Executor, class Function,
             P0443_REQUIRES(ext::is_executor<Executor>::value and
                            is_invocable<Function>::value)
            >
    P0443_ANNOTATION
    executor_operation(Executor&& executor, Function&& function)
      : executor_(std::forward<Executor>(executor)),
        function_(std::forward<Function>(function))
    {}

    P0443_ANNOTATION
    void start()
    {
      P0443_NAMESPACE::execute(executor_, function_);
    }
};


template<class Executor, class Function,
         P0443_REQUIRES(ext::is_executor<remove_cvref_t<Executor>>::value and
                        is_invocable<Function>::value and
                        ext::is_executor_of<remove_cvref_t<Executor>, Function>::value
        )>
P0443_ANNOTATION
executor_operation<remove_cvref_t<Executor>, remove_cvref_t<Function>> make_executor_operation(Executor&& ex, Function&& f)
{
  return {std::forward<Executor>(ex), std::forward<Function>(f)};
}


} // end detail


P0443_NAMESPACE_CLOSE_BRACE

#include "../../epilogue.hpp"

