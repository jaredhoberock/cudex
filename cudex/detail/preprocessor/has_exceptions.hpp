// note that this header file is special and does not use #pragma once

// CUDEX_HAS_EXCEPTIONS indicates whether or not exception support is available.

#ifndef CUDEX_HAS_EXCEPTIONS

#  if defined(__CUDACC__)
#    if !defined(__CUDA_ARCH__)
#      define HAS_EXCEPTIONS __cpp_exceptions
#    else
#      define HAS_EXCEPTIONS 0
#    endif
#  else
#    define HAS_EXCEPTIONS __cpp_exceptions
#  endif

#elif defined(CUDEX_HAS_EXCEPTIONS)
#  undef CUDEX_HAS_EXCEPTIONS
#endif

