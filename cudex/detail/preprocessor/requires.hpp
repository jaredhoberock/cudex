// note that this header file is special and does not use #pragma once

// The CUDEX_REQUIRES() macro may be used in a function template's parameter list
// to simulate Concepts.
//
// For example, to selectively enable a function template only for integer types,
// we could do something like this:
//
//     template<class Integer,
//              CUDEX_REQUIRES(std::is_integral<Integer>::value)
//             >
//     Integer plus_one(Integer x)
//     {
//       return x + 1;
//     }
//

#ifndef CUDEX_REQUIRES

#  define CUDEX_CONCATENATE_IMPL(x, y) x##y

#  define CUDEX_CONCATENATE(x, y) CUDEX_CONCATENATE_IMPL(x, y)

#  define CUDEX_MAKE_UNIQUE(x) CUDEX_CONCATENATE(x, __COUNTER__)

#  define CUDEX_REQUIRES_IMPL(unique_name, ...) bool unique_name = true, typename std::enable_if<(unique_name and __VA_ARGS__)>::type* = nullptr

#  define CUDEX_REQUIRES(...) CUDEX_REQUIRES_IMPL(CUDEX_MAKE_UNIQUE(__deduced_true), __VA_ARGS__)


#  define CUDEX_REQUIRES_DEF_IMPL(unique_name, ...) bool unique_name, typename std::enable_if<(unique_name and __VA_ARGS__)>::type*

#  define CUDEX_REQUIRES_DEF(...) CUDEX_REQUIRES_DEF_IMPL(CUDEX_MAKE_UNIQUE(__deduced_true), __VA_ARGS__)


#elif defined(CUDEX_REQUIRES)

#  ifdef CUDEX_CONCATENATE_IMPL
#    undef CUDEX_CONCATENATE_IMPL
#  endif

#  ifdef CUDEX_CONCATENATE
#    undef CUDEX_CONCATENATE
#  endif

#  ifdef CUDEX_MAKE_UNIQUE
#    undef CUDEX_MAKE_UNIQUE
#  endif

#  ifdef CUDEX_REQUIRES_IMPL
#    undef CUDEX_REQUIRES_IMPL
#  endif

#  ifdef CUDEX_REQUIRES
#    undef CUDEX_REQUIRES
#  endif

#  ifdef CUDEX_REQUIRES_DEF_IMPL
#    undef CUDEX_REQUIRES_DEF_IMPL
#  endif

#  ifdef CUDEX_REQUIRES_DEF
#    undef CUDEX_REQUIRES_DEF
#  endif

#endif

