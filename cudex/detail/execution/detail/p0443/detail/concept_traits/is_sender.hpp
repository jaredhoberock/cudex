#pragma once

#include "../prologue.hpp"

#include <type_traits>
#include "../sender_traits.hpp"
#include "../type_traits/standard_traits.hpp"


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class T, class Enable = void>
struct sender_traits_is_specialized : std::true_type {};

template<class T>
struct sender_traits_is_specialized<
  T,
  typename sender_traits<T>::__unspecialized
> : std::false_type
{};


template<class S>
using is_sender = conjunction<
  std::is_move_constructible<remove_cvref_t<S>>,
  sender_traits_is_specialized<remove_cvref_t<S>>
>;


} // end detail


namespace ext
{


template<class S>
using is_sender = detail::is_sender<S>;
  

} // end ext


P0443_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

