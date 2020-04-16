#pragma once

#include "../prologue.hpp"

#include <exception>
#include <type_traits>
#include <utility>
#include "../type_traits/is_detected.hpp"
#include "../type_traits/is_nothrow_move_or_copy_constructible.hpp"
#include "../type_traits/standard_traits.hpp"
#include "../customization_points/set_done.hpp"
#include "../customization_points/set_error.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class R>
using set_done_t = decltype(P0443_NAMESPACE::set_done(std::declval<R>()));

template<class R, class E>
using set_error_t = decltype(P0443_NAMESPACE::set_error(std::declval<R>(), std::declval<E>()));


template<class R, class E = std::exception_ptr>
using is_receiver = conjunction<
  std::is_move_constructible<remove_cvref_t<R>>,
  is_nothrow_move_or_copy_constructible<remove_cvref_t<R>>,
  is_detected<set_done_t, R>,
  is_detected<set_error_t, R, E>
>;


} // end detail


namespace ext
{


template<class R, class E = std::exception_ptr>
using is_receiver = detail::is_receiver<R,E>;


} // end ext


P0443_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

