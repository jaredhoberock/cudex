// note that this header file is special and does not use #pragma once

// CUDEX_HAS_CUDART indicates whether or not the CUDA Runtime API is available.

#ifndef CUDEX_HAS_CUDART

#  if defined(__CUDACC__)
#    if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__>= 350 && defined(__CUDACC_RDC__))
#      define CUDEX_HAS_CUDART 1
#    else
#      define CUDEX_HAS_CUDART 0
#    endif
#  else
#    define CUDEX_HAS_CUDART __has_include(<cuda_runtime_api.h>)
#  endif

#elif defined(CUDEX_HAS_CUDART)
#  undef CUDEX_HAS_CUDART
#endif

