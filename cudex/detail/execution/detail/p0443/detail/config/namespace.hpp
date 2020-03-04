#if !defined(P0443_NAMESPACE)

// this branch is taken the first time this header is included

#  if defined(P0443_NAMESPACE_OPEN_BRACE) or defined(P0443_NAMESPACE_CLOSE_BRACE)
#    error "Either all of P0443_NAMESPACE, P0443_NAMESPACE_OPEN_BRACE, and P0443_NAMESPACE_CLOSE_BRACE must be defined, or none of them."
#  endif

#  define P0443_NAMESPACE p0443::execution
#  define P0443_NAMESPACE_OPEN_BRACE namespace p0443 { namespace execution {
#  define P0443_NAMESPACE_CLOSE_BRACE } }
#  define P0443_NAMESPACE_NEEDS_UNDEF

#elif defined(P0443_NAMESPACE_NEEDS_UNDEF)

// this branch is taken the second time this header is included

#  undef P0443_NAMESPACE
#  undef P0443_NAMESPACE_OPEN_BRACE
#  undef P0443_NAMESPACE_CLOSE_BRACE
#  undef P0443_NAMESPACE_NEEDS_UNDEF

#elif defined(P0443_NAMESPACE) or defined(P0443_NAMESPACE_OPEN_BRACE) or defined(P0443_NAMESPACE_CLOSE_BRACE)

// this branch is taken the first time this header is included, and the user has misconfigured these namespace-related symbols

#  if !defined(P0443_NAMESPACE) or !defined(P0443_NAMESPACE_OPEN_BRACE) or !defined(P0443_NAMESPACE_CLOSE_BRACE)
#    error "Either all of P0443_NAMESPACE, P0443_NAMESPACE_OPEN_BRACE, and P0443_NAMESPACE_CLOSE_BRACE must be defined, or none of them."
#  endif

#endif

