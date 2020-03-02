// note that this header file is special and does not use #pragma once

#if !defined(CUDEX_NAMESPACE)

// this branch is taken the first time this header is included

#  if defined(CUDEX_NAMESPACE_OPEN_BRACE) or defined(CUDEX_NAMESPACE_CLOSE_BRACE)
#    error "Either all of CUDEX_NAMESPACE, CUDEX_NAMESPACE_OPEN_BRACE, and CUDEX_NAMESPACE_CLOSE_BRACE must be defined, or none of them."
#  endif

#  define CUDEX_NAMESPACE cudex
#  define CUDEX_NAMESPACE_OPEN_BRACE namespace cudex {
#  define CUDEX_NAMESPACE_CLOSE_BRACE }
#  define CUDEX_NAMESPACE_NEEDS_UNDEF

#elif defined(CUDEX_NAMESPACE_NEEDS_UNDEF)

// this branch is taken the second time this header is included

#  undef CUDEX_NAMESPACE
#  undef CUDEX_NAMESPACE_OPEN_BRACE
#  undef CUDEX_NAMESPACE_CLOSE_BRACE
#  undef CUDEX_NAMESPACE_NEEDS_UNDEF

#elif defined(CUDEX_NAMESPACE) or defined(CUDEX_NAMESPACE_OPEN_BRACE) or defined(CUDEX_CLOSE_BRACE)

// this branch is taken the first time this header is included, and the user has misconfigured these namespace-related symbols

#  if !defined(CUDEX_NAMESPACE) or !defined(CUDEX_NAMESPACE_OPEN_BRACE) or !defined(CUDEX_NAMESPACE_CLOSE_BRACE)
#    error "Either all of CUDEX_NAMESPACE, CUDEX_NAMESPACE_OPEN_BRACE, and CUDEX_NAMESPACE_CLOSE_BRACE must be defined, or none of them."
#  endif

#endif

