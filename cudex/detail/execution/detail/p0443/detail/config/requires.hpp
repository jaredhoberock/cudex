// The P0443_REQUIRES() macro may be used in a function template's parameter list
// to simulate Concepts.
//
// For example, to selectively enable a function template only for integer types,
// we could do something like this:
//
//     template<class Integer,
//              P0443_REQUIRES(std::is_integral<Integer>::value)
//             >
//     Integer plus_one(Integer x)
//     {
//       return x + 1;
//     }
//

#ifndef P0443_REQUIRES

#  define P0443_CONCATENATE_IMPL(x, y) x##y

#  define P0443_CONCATENATE(x, y) P0443_CONCATENATE_IMPL(x, y)

#  define P0443_MAKE_UNIQUE(x) P0443_CONCATENATE(x, __COUNTER__)

#  define P0443_REQUIRES_IMPL(unique_name, ...) bool unique_name = true, typename std::enable_if<(unique_name and __VA_ARGS__)>::type* = nullptr

#  define P0443_REQUIRES(...) P0443_REQUIRES_IMPL(P0443_MAKE_UNIQUE(__deduced_true), __VA_ARGS__)

#elif defined(P0443_REQUIRES)

#  ifdef P0443_CONCATENATE_IMPL
#    undef P0443_CONCATENATE_IMPL
#  endif

#  ifdef P0443_CONCATENATE
#    undef P0443_CONCATENATE
#  endif

#  ifdef P0443_MAKE_UNIQUE
#    undef P0443_MAKE_UNIQUE
#  endif

#  ifdef P0443_REQUIRES_IMPL
#    undef P0443_REQUIRES_IMPL
#  endif

#  ifdef P0443_REQUIRES
#    undef P0443_REQUIRES
#  endif

#endif

