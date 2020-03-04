#pragma once

#include "../prologue.hpp"

#include <type_traits>


P0443_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class T>
struct remove_cvref
{
  using type = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
};

template<class T>
using remove_cvref_t = typename remove_cvref<T>::type;


template<class F, class... Args>
struct invoke_result
{
  using type = typename std::result_of<F(Args...)>::type;
};


template<class F, class... Args>
using invoke_result_t = typename invoke_result<F, Args...>::type;


template<class...> struct conjunction;
template<> struct conjunction<> : std::true_type {};
template<class B1> struct conjunction<B1> : B1 {};
template<class B1, class... BN> struct conjunction<B1,BN...> : std::conditional<bool(B1::value), conjunction<BN...>, B1>::type {};


} // end detail


P0443_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

