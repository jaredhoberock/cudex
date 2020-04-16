#pragma once

#include "../../prologue.hpp"

#include <utility>
#include <exception>
#include "../../type_traits/standard_traits.hpp"
#include "../set_done.hpp"
#include "../set_error.hpp"
#include "../set_value.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


// XXX the try/catch below should be guarded with __has_exceptions or similar


template<class R>
class receiver_as_invocable
{
  private:
    using receiver_type = remove_cvref_t<R>;

    // XXX better to replace these members with a __host__ __device__ optional<receiver_type> type when available
    //     once we do that, we can eliminate the P0443_EXEC_CHECK_DISABLE below
    receiver_type r_;
    bool valid_;

  public:
    P0443_EXEC_CHECK_DISABLE
    P0443_ANNOTATION
    explicit receiver_as_invocable(receiver_type&& r)
      try : r_(detail::move_if_noexcept(r)), valid_(true) {}
      catch(...)
      {
        P0443_NAMESPACE::set_error(r, std::current_exception());
      }

    P0443_EXEC_CHECK_DISABLE
    P0443_ANNOTATION
    explicit receiver_as_invocable(const receiver_type& r)
      try : r_(r), valid_(true) {}
      catch(...)
      {
        P0443_NAMESPACE::set_error(r, std::current_exception());
      }

    P0443_EXEC_CHECK_DISABLE
    P0443_ANNOTATION
    receiver_as_invocable(receiver_as_invocable&& other)
      try : r_(detail::move_if_noexcept(other.r_)), valid_(other.valid_)
      {
        other.valid_ = false;
      }
      catch(...)
      {
        P0443_NAMESPACE::set_error(other.receiver_, std::current_exception());
      }

    P0443_EXEC_CHECK_DISABLE
    P0443_ANNOTATION
    ~receiver_as_invocable()
    {
      if(valid_)
      {
        P0443_NAMESPACE::set_done(r_);
      }
    }

    void operator()()
    {
      try
      {
        P0443_NAMESPACE::set_value(r_);
      }
      catch(...)
      {
        P0443_NAMESPACE::set_error(r_, std::current_exception());
      }

      valid_ = false;
    }
};


template<class R>
P0443_ANNOTATION
receiver_as_invocable<R&&> as_invocable(R&& r)
{
  return receiver_as_invocable<R&&>{std::forward<R>(r)};
}


} // end detail


P0443_NAMESPACE_CLOSE_BRACE

#include "../../epilogue.hpp"

