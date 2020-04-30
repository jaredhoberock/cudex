// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <stddef.h> // XXX instead of <cstddef> to WAR clang issue
#include <type_traits>
#include <utility> // <utility> declares std::tuple_element et al. for us

// allow the user to define an annotation to apply to these functions
#ifndef TUPLE_ANNOTATION
#  if defined(__CUDACC__) && !(defined(__CUDA__) && defined(__clang__))
#    define TUPLE_ANNOTATION __host__ __device__
#  else
#    define TUPLE_ANNOTATION
#  endif // c++ <= 2011
#endif


// define the incantation to silence nvcc errors concerning __host__ __device__ functions
#if defined(__CUDACC__) && !(defined(__CUDA__) && defined(__clang__)) and !defined(__NVCOMPILER_CUDA__)
#  define TUPLE_EXEC_CHECK_DISABLE \
#  pragma nv_exec_check_disable
#else
#  define TUPLE_EXEC_CHECK_DISABLE
#endif

// allow the user to define a namespace for these functions
#if !defined(TUPLE_NAMESPACE)

#  if defined(TUPLE_NAMESPACE_OPEN_BRACE) or defined(TUPLE_NAMESPACE_CLOSE_BRACE)
#    error "All or none of TUPLE_NAMESPACE, TUPLE_NAMESPACE_OPEN_BRACE, and TUPLE_NAMESPACE_CLOSE_BRACE must be defined."
#  endif

#  define TUPLE_NAMESPACE std
#  define TUPLE_NAMESPACE_OPEN_BRACE namespace std {
#  define TUPLE_NAMESPACE_CLOSE_BRACE }
#  define TUPLE_NAMESPACE_NEEDS_UNDEF

#else

#  if !defined(TUPLE_NAMESPACE_OPEN_BRACE) or !defined(TUPLE_NAMESPACE_CLOSE_BRACE)
#    error "All or none of TUPLE_NAMESPACE, TUPLE_NAMESPACE_OPEN_BRACE, and TUPLE_NAMESPACE_CLOSE_BRACE must be defined."
#  endif

#endif


// allow the user to define a singly-nested namespace for private implementation details
#if !defined(TUPLE_DETAIL_NAMESPACE)
#  define TUPLE_DETAIL_NAMESPACE detail
#  define TUPLE_DETAIL_NAMESPACE_NEEDS_UNDEF
#endif


TUPLE_NAMESPACE_OPEN_BRACE

// first declare tuple 
template<class... Types> class tuple;

TUPLE_NAMESPACE_CLOSE_BRACE


// specializations of stuff in std come before their use
namespace std
{


template<size_t i>
class tuple_element<i, TUPLE_NAMESPACE::tuple<>> {};


template<class Type1, class... Types>
class tuple_element<0, TUPLE_NAMESPACE::tuple<Type1,Types...>>
{
  public:
    using type = Type1;
};


template<size_t i, class Type1, class... Types>
class tuple_element<i, TUPLE_NAMESPACE::tuple<Type1,Types...>>
{
  public:
    using type = typename tuple_element<i - 1, TUPLE_NAMESPACE::tuple<Types...>>::type;
};


template<class... Types>
class tuple_size<TUPLE_NAMESPACE::tuple<Types...>>
  : public std::integral_constant<size_t, sizeof...(Types)>
{};


} // end std


TUPLE_NAMESPACE_OPEN_BRACE


namespace TUPLE_DETAIL_NAMESPACE
{

// define variadic "and" operator 
template <typename... Conditions>
  struct tuple_and;

template<>
  struct tuple_and<>
    : public std::true_type
{
};

template <typename Condition, typename... Conditions>
  struct tuple_and<Condition, Conditions...>
    : public std::integral_constant<
        bool,
        Condition::value && tuple_and<Conditions...>::value>
{
};

// XXX this implementation is based on Howard Hinnant's "tuple leaf" construction in libcxx


// define index sequence in case it is missing
// prefix this stuff with "tuple" to avoid collisions with other implementations
template<size_t... I> struct tuple_index_sequence {};

template<size_t Start, typename Indices, size_t End>
struct tuple_make_index_sequence_impl;

template<size_t Start, size_t... Indices, size_t End>
struct tuple_make_index_sequence_impl<
  Start,
  tuple_index_sequence<Indices...>, 
  End
>
{
  typedef typename tuple_make_index_sequence_impl<
    Start + 1,
    tuple_index_sequence<Indices..., Start>,
    End
  >::type type;
};

template<size_t End, size_t... Indices>
struct tuple_make_index_sequence_impl<End, tuple_index_sequence<Indices...>, End>
{
  typedef tuple_index_sequence<Indices...> type;
};

template<size_t N>
using tuple_make_index_sequence = typename tuple_make_index_sequence_impl<0, tuple_index_sequence<>, N>::type;


template<class T>
struct use_empty_base_class_optimization
  : std::integral_constant<
      bool,
      std::is_empty<T>::value
#if __cplusplus >= 201402L
      && !std::is_final<T>::value
#endif
    >
{};


template<class T>
using tuple_leaf_base_case = typename std::conditional<
  // the empty T case is case 0
  std::is_empty<T>::value,
  std::integral_constant<int,0>,

  typename std::conditional<
    std::is_trivially_copyable<T>::value,

    // the trivially copyable T case is case 1
    std::integral_constant<int,1>,

    // the general case is case 2
    std::integral_constant<int,2>
  >::type
>::type;


template<class T, int = tuple_leaf_base_case<T>::value>
class tuple_leaf_base;


// This specialization handles the case when T is empty
// The reason we don't inherit from T to get the empty base class optimization
// is because the inheritance creates ambiguity in cases like
//
//     tuple<empty, tuple<empty>> or
//
//     In this example, the outer tuple would end up inheriting from tuple_leaf<0,empty> twice
//
//  Note that if T's ctors, dtor, or assignments have effects, this implementation ends up ignoring them
//  in favor of trival copyability
//
template<class T>
class tuple_leaf_base<T,0>
{
  public:
    TUPLE_EXEC_CHECK_DISABLE
    tuple_leaf_base() = default;

    TUPLE_EXEC_CHECK_DISABLE
    tuple_leaf_base(const tuple_leaf_base&) = default;

    TUPLE_EXEC_CHECK_DISABLE
    tuple_leaf_base(tuple_leaf_base&&) = default;

    TUPLE_EXEC_CHECK_DISABLE
    template<class U>
    TUPLE_ANNOTATION
    tuple_leaf_base(U&& arg)
    {
      // invoke T's ctor only for its effects
      new(&mutable_get()) T{std::forward<U>(arg)};
    }

    TUPLE_EXEC_CHECK_DISABLE
    ~tuple_leaf_base() = default;

    TUPLE_EXEC_CHECK_DISABLE
    tuple_leaf_base& operator=(const tuple_leaf_base&) = default;

    TUPLE_EXEC_CHECK_DISABLE
    tuple_leaf_base& operator=(tuple_leaf_base&&) = default;

    TUPLE_ANNOTATION
    const T& const_get() const
    {
      return *reinterpret_cast<const T*>(this);
    }
  
    TUPLE_ANNOTATION
    T& mutable_get()
    {
      return *reinterpret_cast<T*>(this);
    }

  private:
    // exposition only
    // T value_;
};




// This specialization handles the case when T is trivially-copyable
template<class T>
class tuple_leaf_base<T,1>
{
  public:
    TUPLE_EXEC_CHECK_DISABLE
    TUPLE_ANNOTATION
    tuple_leaf_base() : val_{} {}

    TUPLE_EXEC_CHECK_DISABLE
    tuple_leaf_base(const tuple_leaf_base&) = default;

    TUPLE_EXEC_CHECK_DISABLE
    tuple_leaf_base(tuple_leaf_base&&) = default;

    TUPLE_EXEC_CHECK_DISABLE
    template<class U>
    TUPLE_ANNOTATION
    tuple_leaf_base(U&& arg) : val_(std::forward<U>(arg)) {}

    TUPLE_EXEC_CHECK_DISABLE
    ~tuple_leaf_base() = default;

    TUPLE_EXEC_CHECK_DISABLE
    tuple_leaf_base& operator=(const tuple_leaf_base&) = default;

    TUPLE_EXEC_CHECK_DISABLE
    tuple_leaf_base& operator=(tuple_leaf_base&&) = default;

    TUPLE_ANNOTATION
    const T& const_get() const
    {
      return val_;
    }

    TUPLE_ANNOTATION
    T& mutable_get()
    {
      return val_;
    }

  private:
    T val_;
};


// this specialization handles the general case
template<class T>
class tuple_leaf_base<T,2>
{
  public:
    TUPLE_EXEC_CHECK_DISABLE
    tuple_leaf_base() = default;

    TUPLE_EXEC_CHECK_DISABLE
    tuple_leaf_base(const tuple_leaf_base&) = default;

    TUPLE_EXEC_CHECK_DISABLE
    tuple_leaf_base(tuple_leaf_base&&) = default;

    TUPLE_EXEC_CHECK_DISABLE
    template<class U,
             class = typename std::enable_if<
               std::is_constructible<T,U>::value
             >::type>
    TUPLE_ANNOTATION
    tuple_leaf_base(U&& arg) : val_{std::forward<U>(arg)} {}

    TUPLE_EXEC_CHECK_DISABLE
    ~tuple_leaf_base() = default;

    TUPLE_EXEC_CHECK_DISABLE
    tuple_leaf_base& operator=(const tuple_leaf_base& other)
    {
      val_ = other.const_get();
      return *this;
    }

    TUPLE_EXEC_CHECK_DISABLE
    template<class U, int I,
             class = typename std::enable_if<
               std::is_assignable<T,const U&>::value
             >::type>
    TUPLE_ANNOTATION
    tuple_leaf_base& operator=(const tuple_leaf_base<U,I>& other)
    {
      val_ = other.const_get();
      return *this;
    }

    TUPLE_EXEC_CHECK_DISABLE
    tuple_leaf_base& operator=(tuple_leaf_base&& other)
    {
      val_ = std::forward<T>(other.mutable_get());
      return *this;
    }

    TUPLE_EXEC_CHECK_DISABLE
    template<class U, int I,
             class = typename std::enable_if<
               std::is_assignable<T,U&&>::value
             >::type>
    TUPLE_ANNOTATION
    tuple_leaf_base& operator=(tuple_leaf_base<U,I>&& other)
    {
      val_ = std::forward<U>(other.mutable_get());
      return *this;
    }

    TUPLE_ANNOTATION
    const T& const_get() const
    {
      return val_;
    }

    TUPLE_ANNOTATION
    T& mutable_get()
    {
      return val_;
    }

  private:
    T val_;
};


template<size_t I, class T>
class tuple_leaf : public tuple_leaf_base<T>
{
  private:
    using super_t = tuple_leaf_base<T>;

  public:
    tuple_leaf() = default;

    template<class U,
             class = typename std::enable_if<
               std::is_constructible<T,U>::value
             >::type>
    TUPLE_ANNOTATION
    tuple_leaf(U&& arg) : super_t(std::forward<U>(arg)) {}

    tuple_leaf(const tuple_leaf&) = default;

    tuple_leaf(tuple_leaf&&) = default;

    template<class U,
             class = typename std::enable_if<
               std::is_constructible<T,const U&>::value
             >::type>
    TUPLE_ANNOTATION
    tuple_leaf(const tuple_leaf<I,U>& other) : super_t(other.const_get()) {}
    
    tuple_leaf& operator=(const tuple_leaf&) = default;

    tuple_leaf& operator=(tuple_leaf&&) = default;

    // converting move-constructor
    // note the use of std::forward<U> here to allow construction of T from U&&
    template<class U,
             class = typename std::enable_if<
               std::is_constructible<T,U&&>::value
             >::type>
    TUPLE_ANNOTATION
    tuple_leaf(tuple_leaf<I,U>&& other) : super_t(std::forward<U>(other.mutable_get())) {}


    TUPLE_EXEC_CHECK_DISABLE
    template<class U,
             class = typename std::enable_if<
               std::is_assignable<T,U>::value
             >::type>
    TUPLE_ANNOTATION
    tuple_leaf& operator=(const tuple_leaf<I,U>& other)
    {
      this->mutable_get() = other.const_get();
      return *this;
    }

    TUPLE_EXEC_CHECK_DISABLE
    template<class U,
             class = typename std::enable_if<
               std::is_assignable<T,U&&>::value
             >::type>
    TUPLE_ANNOTATION
    tuple_leaf& operator=(tuple_leaf<I,U>&& other)
    {
      this->mutable_get() = std::forward<U>(other.mutable_get());
      return *this;
    }

    TUPLE_EXEC_CHECK_DISABLE
    TUPLE_ANNOTATION
    int swap(tuple_leaf& other)
    {
      using std::swap;
      swap(this->mutable_get(), other.mutable_get());
      return 0;
    }
};

template<class... Args>
struct tuple_type_list {};

template<size_t i, class... Args>
struct tuple_type_at_impl;

template<size_t i, class Arg0, class... Args>
struct tuple_type_at_impl<i, Arg0, Args...>
{
  using type = typename tuple_type_at_impl<i-1, Args...>::type;
};

template<class Arg0, class... Args>
struct tuple_type_at_impl<0, Arg0,Args...>
{
  using type = Arg0;
};

template<size_t i, class... Args>
using tuple_type_at = typename tuple_type_at_impl<i,Args...>::type;

template<class IndexSequence, class... Args>
class tuple_base;

template<size_t... I, class... Types>
class tuple_base<tuple_index_sequence<I...>, Types...>
  : public tuple_leaf<I,Types>...
{
  public:
    using leaf_types = tuple_type_list<tuple_leaf<I,Types>...>;

    tuple_base() = default;

    tuple_base(const tuple_base&) = default;

    tuple_base(tuple_base&&) = default;


    TUPLE_ANNOTATION
    tuple_base(const Types&... args)
      : tuple_leaf<I,Types>(args)...
    {}


    template<class... UTypes,
             class = typename std::enable_if<
               (sizeof...(Types) == sizeof...(UTypes)) &&
               tuple_and<
                 std::is_constructible<Types,UTypes&&>...
               >::value
             >::type>
    TUPLE_ANNOTATION
    explicit tuple_base(UTypes&&... args)
      : tuple_leaf<I,Types>(std::forward<UTypes>(args))...
    {}


    template<class... UTypes,
             class = typename std::enable_if<
               (sizeof...(Types) == sizeof...(UTypes)) &&
               tuple_and<
                 std::is_constructible<Types,const UTypes&>...
                >::value
             >::type>
    TUPLE_ANNOTATION
    tuple_base(const tuple_base<tuple_index_sequence<I...>,UTypes...>& other)
      : tuple_leaf<I,Types>(other.template const_leaf<I>())...
    {}


    template<class... UTypes,
             class = typename std::enable_if<
               (sizeof...(Types) == sizeof...(UTypes)) &&
               tuple_and<
                 std::is_constructible<Types,UTypes&&>...
                >::value
             >::type>
    TUPLE_ANNOTATION
    tuple_base(tuple_base<tuple_index_sequence<I...>,UTypes...>&& other)
      : tuple_leaf<I,Types>(std::move(other.template mutable_leaf<I>()))...
    {}


    //template<class... UTypes,
    //         class = typename std::enable_if<
    //           (sizeof...(Types) == sizeof...(UTypes)) &&
    //           tuple_and<
    //             std::is_constructible<Types,const UTypes&>...
    //            >::value
    //         >::type>
    //TUPLE_ANNOTATION
    //tuple_base(const std::tuple<UTypes...>& other)
    //  : tuple_base{std::get<I>(other)...}
    //{}


    tuple_base& operator=(const tuple_base&) = default;

    tuple_base& operator=(tuple_base&&) = default;


    template<class... UTypes,
             class = typename std::enable_if<
               (sizeof...(Types) == sizeof...(UTypes)) &&
               tuple_and<
                 std::is_assignable<Types,const UTypes&>...
                >::value
             >::type>
    TUPLE_ANNOTATION
    tuple_base& operator=(const tuple_base<tuple_index_sequence<I...>,UTypes...>& other)
    {
      swallow((mutable_leaf<I>() = other.template const_leaf<I>())...);
      return *this;
    }

    template<class... UTypes,
             class = typename std::enable_if<
               (sizeof...(Types) == sizeof...(UTypes)) &&
               tuple_and<
                 std::is_assignable<Types,UTypes&&>...
               >::value
             >::type>
    TUPLE_ANNOTATION
    tuple_base& operator=(tuple_base<tuple_index_sequence<I...>,UTypes...>&& other)
    {
      swallow((mutable_leaf<I>() = std::move(other.template mutable_leaf<I>()))...);
      return *this;
    }

    template<class UType1, class UType2,
             class = typename std::enable_if<
               (sizeof...(Types) == 2) &&
               tuple_and<
                 std::is_assignable<tuple_type_at<                            0,Types...>,const UType1&>,
                 std::is_assignable<tuple_type_at<sizeof...(Types) == 2 ? 1 : 0,Types...>,const UType2&>
               >::value
             >::type>
    TUPLE_ANNOTATION
    tuple_base& operator=(const std::pair<UType1,UType2>& p)
    {
      mutable_get<0>() = p.first;
      mutable_get<1>() = p.second;
      return *this;
    }

    template<class UType1, class UType2,
             class = typename std::enable_if<
               (sizeof...(Types) == 2) &&
               tuple_and<
                 std::is_assignable<tuple_type_at<                            0,Types...>,UType1&&>,
                 std::is_assignable<tuple_type_at<sizeof...(Types) == 2 ? 1 : 0,Types...>,UType2&&>
               >::value
             >::type>
    TUPLE_ANNOTATION
    tuple_base& operator=(std::pair<UType1,UType2>&& p)
    {
      mutable_get<0>() = std::move(p.first);
      mutable_get<1>() = std::move(p.second);
      return *this;
    }

    template<size_t i>
    TUPLE_ANNOTATION
    const tuple_leaf<i,tuple_type_at<i,Types...>>& const_leaf() const
    {
      return *this;
    }

    template<size_t i>
    TUPLE_ANNOTATION
    tuple_leaf<i,tuple_type_at<i,Types...>>& mutable_leaf()
    {
      return *this;
    }

    template<size_t i>
    TUPLE_ANNOTATION
    tuple_leaf<i,tuple_type_at<i,Types...>>&& move_leaf() &&
    {
      return std::move(*this);
    }

    TUPLE_ANNOTATION
    void swap(tuple_base& other)
    {
      swallow(tuple_leaf<I,Types>::swap(other)...);
    }

    template<size_t i>
    TUPLE_ANNOTATION
    const tuple_type_at<i,Types...>& const_get() const
    {
      return const_leaf<i>().const_get();
    }

    template<size_t i>
    TUPLE_ANNOTATION
    tuple_type_at<i,Types...>& mutable_get()
    {
      return mutable_leaf<i>().mutable_get();
    }

    //// enable conversion to Tuple-like things
    //template<class... UTypes,
    //         class = typename std::enable_if<
    //           (sizeof...(Types) == sizeof...(UTypes)) &&
    //           tuple_and<
    //             std::is_constructible<Types,const UTypes&>...
    //            >::value
    //         >::type>
    //TUPLE_ANNOTATION
    //operator std::tuple<UTypes...> () const
    //{
    //  return std::tuple<UTypes...>(const_get<I>()...);
    //}

  private:
    template<class... Args>
    TUPLE_ANNOTATION
    static void swallow(Args&&...) {}
};


} // end TUPLE_DETAIL_NAMESPACE


template<size_t i, class... UTypes>
TUPLE_ANNOTATION
typename std::tuple_element<i, TUPLE_NAMESPACE::tuple<UTypes...>>::type &
  get(TUPLE_NAMESPACE::tuple<UTypes...>& t)
{
  return t.template mutable_get<i>();
}


template<size_t i, class... UTypes>
TUPLE_ANNOTATION
const typename std::tuple_element<i, TUPLE_NAMESPACE::tuple<UTypes...>>::type &
  get(const TUPLE_NAMESPACE::tuple<UTypes...>& t)
{
  return t.template const_get<i>();
}


template<size_t i, class... UTypes>
TUPLE_ANNOTATION
typename std::tuple_element<i, TUPLE_NAMESPACE::tuple<UTypes...>>::type &&
  get(TUPLE_NAMESPACE::tuple<UTypes...>&& t)
{
  using type = typename std::tuple_element<i, TUPLE_NAMESPACE::tuple<UTypes...>>::type;

  auto&& leaf = static_cast<TUPLE_NAMESPACE::TUPLE_DETAIL_NAMESPACE::tuple_leaf<i,type>&&>(t.base());

  return static_cast<type&&>(leaf.mutable_get());
}


template<class... Types>
class tuple : private TUPLE_DETAIL_NAMESPACE::tuple_base<TUPLE_DETAIL_NAMESPACE::tuple_make_index_sequence<sizeof...(Types)>, Types...>
{
  private:
    using base_type = TUPLE_DETAIL_NAMESPACE::tuple_base<TUPLE_DETAIL_NAMESPACE::tuple_make_index_sequence<sizeof...(Types)>, Types...>;

    TUPLE_ANNOTATION
    base_type& base()
    {
      return *this;
    }

    TUPLE_ANNOTATION
    const base_type& base() const
    {
      return *this;
    }

  public:
    tuple() = default;

    tuple(const tuple&) = default;

    tuple(tuple&&) = default;

    TUPLE_ANNOTATION
    explicit tuple(const Types&... args)
      : base_type{args...}
    {}

    template<class... UTypes,
             class = typename std::enable_if<
               (sizeof...(Types) == sizeof...(UTypes)) &&
               TUPLE_DETAIL_NAMESPACE::tuple_and<
                 std::is_constructible<Types,UTypes&&>...
               >::value
             >::type>
    TUPLE_ANNOTATION
    explicit tuple(UTypes&&... args)
      : base_type{std::forward<UTypes>(args)...}
    {}

    template<class... UTypes,
             class = typename std::enable_if<
               (sizeof...(Types) == sizeof...(UTypes)) &&
                 TUPLE_DETAIL_NAMESPACE::tuple_and<
                   std::is_constructible<Types,const UTypes&>...
                 >::value
             >::type>
    TUPLE_ANNOTATION
    tuple(const tuple<UTypes...>& other)
      : base_type{other.base()}
    {}

    template<class... UTypes,
             class = typename std::enable_if<
               (sizeof...(Types) == sizeof...(UTypes)) &&
                 TUPLE_DETAIL_NAMESPACE::tuple_and<
                   std::is_constructible<Types,UTypes&&>...
                 >::value
             >::type>
    TUPLE_ANNOTATION
    tuple(tuple<UTypes...>&& other)
      : base_type{std::move(other.base())}
    {}

    template<class UType1, class UType2,
             class = typename std::enable_if<
               (sizeof...(Types) == 2) &&
               TUPLE_DETAIL_NAMESPACE::tuple_and<
                 std::is_constructible<TUPLE_DETAIL_NAMESPACE::tuple_type_at<                            0,Types...>,const UType1&>,
                 std::is_constructible<TUPLE_DETAIL_NAMESPACE::tuple_type_at<sizeof...(Types) == 2 ? 1 : 0,Types...>,const UType2&>
               >::value
             >::type>
    TUPLE_ANNOTATION
    tuple(const std::pair<UType1,UType2>& p)
      : base_type{p.first, p.second}
    {}

    template<class UType1, class UType2,
             class = typename std::enable_if<
               (sizeof...(Types) == 2) &&
               TUPLE_DETAIL_NAMESPACE::tuple_and<
                 std::is_constructible<TUPLE_DETAIL_NAMESPACE::tuple_type_at<                            0,Types...>,UType1&&>,
                 std::is_constructible<TUPLE_DETAIL_NAMESPACE::tuple_type_at<sizeof...(Types) == 2 ? 1 : 0,Types...>,UType2&&>
               >::value
             >::type>
    TUPLE_ANNOTATION
    tuple(std::pair<UType1,UType2>&& p)
      : base_type{std::move(p.first), std::move(p.second)}
    {}

    //template<class... UTypes,
    //         class = typename std::enable_if<
    //           (sizeof...(Types) == sizeof...(UTypes)) &&
    //             TUPLE_DETAIL_NAMESPACE::tuple_and<
    //               std::is_constructible<Types,const UTypes&>...
    //             >::value
    //         >::type>
    //TUPLE_ANNOTATION
    //tuple(const std::tuple<UTypes...>& other)
    //  : base_type{other}
    //{}

    tuple& operator=(const tuple&) = default;

    tuple& operator=(tuple&&) = default;

    // XXX needs enable_if
    template<class... UTypes>
    TUPLE_ANNOTATION
    tuple& operator=(const tuple<UTypes...>& other)
    {
      base().operator=(std::move(other.base()));
      return *this;
    }

    // XXX needs enable_if
    template<class... UTypes>
    TUPLE_ANNOTATION
    tuple& operator=(tuple<UTypes...>&& other)
    {
      base().operator=(other.base());
      return *this;
    }

    template<class UType1, class UType2,
             class = typename std::enable_if<
               (sizeof...(Types) == 2) &&
               TUPLE_DETAIL_NAMESPACE::tuple_and<
                 std::is_assignable<TUPLE_DETAIL_NAMESPACE::tuple_type_at<                            0,Types...>,const UType1&>,
                 std::is_assignable<TUPLE_DETAIL_NAMESPACE::tuple_type_at<sizeof...(Types) == 2 ? 1 : 0,Types...>,const UType2&>
               >::value
             >::type>
    TUPLE_ANNOTATION
    tuple& operator=(const std::pair<UType1,UType2>& p)
    {
      base().operator=(p);
      return *this;
    }

    template<class UType1, class UType2,
             class = typename std::enable_if<
               (sizeof...(Types) == 2) &&
               TUPLE_DETAIL_NAMESPACE::tuple_and<
                 std::is_assignable<TUPLE_DETAIL_NAMESPACE::tuple_type_at<                            0,Types...>,UType1&&>,
                 std::is_assignable<TUPLE_DETAIL_NAMESPACE::tuple_type_at<sizeof...(Types) == 2 ? 1 : 0,Types...>,UType2&&>
               >::value
             >::type>
    TUPLE_ANNOTATION
    tuple& operator=(std::pair<UType1,UType2>&& p)
    {
      base().operator=(std::move(p));
      return *this;
    }

    TUPLE_ANNOTATION
    void swap(tuple& other)
    {
      base().swap(other.base());
    }

    //// enable conversion to Tuple-like things
    //template<class... UTypes,
    //         class = typename std::enable_if<
    //           (sizeof...(Types) == sizeof...(UTypes)) &&
    //           TUPLE_DETAIL_NAMESPACE::tuple_and<
    //             std::is_constructible<Types,const UTypes&>...
    //            >::value
    //         >::type>
    //TUPLE_ANNOTATION
    //operator std::tuple<UTypes...> () const
    //{
    //  return static_cast<std::tuple<UTypes...>>(base());
    //}

  private:
    template<class... UTypes>
    friend class tuple;

    template<size_t i>
    TUPLE_ANNOTATION
    const typename std::tuple_element<i,tuple>::type& const_get() const
    {
      return base().template const_get<i>();
    }

    template<size_t i>
    TUPLE_ANNOTATION
    typename std::tuple_element<i,tuple>::type& mutable_get()
    {
      return base().template mutable_get<i>();
    }

  public:
    template<size_t i, class... UTypes>
    friend TUPLE_ANNOTATION
    typename std::tuple_element<i, TUPLE_NAMESPACE::tuple<UTypes...>>::type &
    get(TUPLE_NAMESPACE::tuple<UTypes...>& t);


    template<size_t i, class... UTypes>
    friend TUPLE_ANNOTATION
    const typename std::tuple_element<i, TUPLE_NAMESPACE::tuple<UTypes...>>::type &
    get(const TUPLE_NAMESPACE::tuple<UTypes...>& t);


    template<size_t i, class... UTypes>
    friend TUPLE_ANNOTATION
    typename std::tuple_element<i, TUPLE_NAMESPACE::tuple<UTypes...>>::type &&
    get(TUPLE_NAMESPACE::tuple<UTypes...>&& t);
};


template<>
class tuple<>
{
  public:
    TUPLE_ANNOTATION
    void swap(tuple&){}
};


template<class... Types>
TUPLE_ANNOTATION
void swap(tuple<Types...>& a, tuple<Types...>& b)
{
  a.swap(b);
}


template<class... Types>
TUPLE_ANNOTATION
tuple<typename std::decay<Types>::type...> make_tuple(Types&&... args)
{
  return tuple<typename std::decay<Types>::type...>(std::forward<Types>(args)...);
}


template<class... Types>
TUPLE_ANNOTATION
tuple<Types&...> tie(Types&... args)
{
  return tuple<Types&...>(args...);
}


template<class... Args>
TUPLE_ANNOTATION
TUPLE_NAMESPACE::tuple<Args&&...> forward_as_tuple(Args&&... args)
{
  return TUPLE_NAMESPACE::tuple<Args&&...>(std::forward<Args>(args)...);
}


namespace TUPLE_DETAIL_NAMESPACE
{


struct tuple_ignore_t
{
  template<class T>
  TUPLE_ANNOTATION
  const tuple_ignore_t operator=(T&&) const
  {
    return *this;
  }
};


} // end TUPLE_DETAIL_NAMESPACE


constexpr TUPLE_DETAIL_NAMESPACE::tuple_ignore_t ignore{};


namespace TUPLE_DETAIL_NAMESPACE
{


template<size_t I, class T, class... Types>
struct tuple_find_exactly_one_impl;


template<size_t I, class T, class U, class... Types>
struct tuple_find_exactly_one_impl<I,T,U,Types...> : tuple_find_exactly_one_impl<I+1, T, Types...> {};


template<size_t I, class T, class... Types>
struct tuple_find_exactly_one_impl<I,T,T,Types...> : std::integral_constant<size_t, I>
{
  static_assert(tuple_find_exactly_one_impl<I,T,Types...>::value == -1, "type can only occur once in type list");
};


template<size_t I, class T>
struct tuple_find_exactly_one_impl<I,T> : std::integral_constant<int, -1> {};


template<class T, class... Types>
struct tuple_find_exactly_one : tuple_find_exactly_one_impl<0,T,Types...>
{
  static_assert(int(tuple_find_exactly_one::value) != -1, "type not found in type list");
};


} // end TUPLE_DETAIL_NAMESPACE



template<class T, class... Types>
TUPLE_ANNOTATION
T& get(TUPLE_NAMESPACE::tuple<Types...>& t)
{
  return TUPLE_NAMESPACE::get<TUPLE_NAMESPACE::TUPLE_DETAIL_NAMESPACE::tuple_find_exactly_one<T,Types...>::value>(t);
}


template<class T, class... Types>
TUPLE_ANNOTATION
const T& get(const TUPLE_NAMESPACE::tuple<Types...>& t)
{
  return TUPLE_NAMESPACE::get<TUPLE_NAMESPACE::TUPLE_DETAIL_NAMESPACE::tuple_find_exactly_one<T,Types...>::value>(t);
}


template<class T, class... Types>
TUPLE_ANNOTATION
T&& get(TUPLE_NAMESPACE::tuple<Types...>&& t)
{
  return TUPLE_NAMESPACE::get<TUPLE_NAMESPACE::TUPLE_DETAIL_NAMESPACE::tuple_find_exactly_one<T,Types...>::value>(std::move(t));
}


// relational operators
namespace TUPLE_DETAIL_NAMESPACE
{


TUPLE_ANNOTATION
inline bool tuple_all()
{
  return true;
}


TUPLE_ANNOTATION
inline bool tuple_all(bool t)
{
  return t;
}


template<typename... Bools>
TUPLE_ANNOTATION
bool tuple_all(bool t, Bools... ts)
{
  return t && TUPLE_DETAIL_NAMESPACE::tuple_all(ts...);
}


} // end TUPLE_DETAIL_NAMESPACE


template<class... TTypes, class... UTypes>
TUPLE_ANNOTATION
bool operator==(const tuple<TTypes...>& t, const tuple<UTypes...>& u);


namespace TUPLE_DETAIL_NAMESPACE
{


template<class... TTypes, class... UTypes, size_t... I>
TUPLE_ANNOTATION
bool tuple_eq(const tuple<TTypes...>& t, const tuple<UTypes...>& u, TUPLE_DETAIL_NAMESPACE::tuple_index_sequence<I...>)
{
  return TUPLE_DETAIL_NAMESPACE::tuple_all((TUPLE_NAMESPACE::get<I>(t) == TUPLE_NAMESPACE::get<I>(u))...);
}


} // end TUPLE_DETAIL_NAMESPACE


template<class... TTypes, class... UTypes>
TUPLE_ANNOTATION
bool operator==(const tuple<TTypes...>& t, const tuple<UTypes...>& u)
{
  return TUPLE_DETAIL_NAMESPACE::tuple_eq(t, u, TUPLE_DETAIL_NAMESPACE::tuple_make_index_sequence<sizeof...(TTypes)>{});
}


template<class... TTypes, class... UTypes>
TUPLE_ANNOTATION
bool operator<(const tuple<TTypes...>& t, const tuple<UTypes...>& u);


namespace TUPLE_DETAIL_NAMESPACE
{


template<class... TTypes, class... UTypes>
TUPLE_ANNOTATION
bool tuple_lt(const tuple<TTypes...>&, const tuple<UTypes...>&, tuple_index_sequence<>)
{
  return false;
}


template<size_t I, class... TTypes, class... UTypes, size_t... Is>
TUPLE_ANNOTATION
bool tuple_lt(const tuple<TTypes...>& t, const tuple<UTypes...>& u, tuple_index_sequence<I, Is...>)
{
  return (TUPLE_NAMESPACE::get<I>(t) < TUPLE_NAMESPACE::get<I>(u)
          or (!(TUPLE_NAMESPACE::get<I>(u) < TUPLE_NAMESPACE::get<I>(t))
          and TUPLE_DETAIL_NAMESPACE::tuple_lt(t, u, typename tuple_make_index_sequence_impl<I+1, tuple_index_sequence<>, sizeof...(TTypes)>::type{})));
}


} // end TUPLE_DETAIL_NAMESPACE


template<class... TTypes, class... UTypes>
TUPLE_ANNOTATION
bool operator<(const tuple<TTypes...>& t, const tuple<UTypes...>& u)
{
  return TUPLE_DETAIL_NAMESPACE::tuple_lt(t, u, TUPLE_DETAIL_NAMESPACE::tuple_make_index_sequence<sizeof...(TTypes)>{});
}


template<class... TTypes, class... UTypes>
TUPLE_ANNOTATION
bool operator!=(const tuple<TTypes...>& t, const tuple<UTypes...>& u)
{
  return !(t == u);
}


template<class... TTypes, class... UTypes>
TUPLE_ANNOTATION
bool operator>(const tuple<TTypes...>& t, const tuple<UTypes...>& u)
{
  return u < t;
}


template<class... TTypes, class... UTypes>
TUPLE_ANNOTATION
bool operator<=(const tuple<TTypes...>& t, const tuple<UTypes...>& u)
{
  return !(u < t);
}


template<class... TTypes, class... UTypes>
TUPLE_ANNOTATION
bool operator>=(const tuple<TTypes...>& t, const tuple<UTypes...>& u)
{
  return !(t < u);
}


TUPLE_NAMESPACE_CLOSE_BRACE


#ifdef TUPLE_ANNOTATION_NEEDS_UNDEF
#  undef TUPLE_ANNOTATION
#  undef TUPLE_ANNOTATION_NEEDS_UNDEF
#endif

#ifdef TUPLE_NAMESPACE_NEEDS_UNDEF
#  undef TUPLE_NAMESPACE
#  undef TUPLE_NAMESPACE_OPEN_BRACE
#  undef TUPLE_NAMESPACE_CLOSE_BRACE
#  undef TUPLE_NAMESPACE_NEEDS_UNDEF
#endif

#ifdef TUPLE_DETAIL_NAMESPACE_NEEDS_UNDEF
#  undef TUPLE_DETAIL_NAMESPACE
#  undef TUPLE_DETAIL_NAMESPACE_NEEDS_UNDEF
#endif

#undef TUPLE_EXEC_CHECK_DISABLE

