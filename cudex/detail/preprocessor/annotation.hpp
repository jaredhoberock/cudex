// note that this header file is special and does not use #pragma once

// CUDEX_ANNOTATION expands to __host__ __device__ when encountered by a
// CUDA-capable compiler

#if !defined(CUDEX_ANNOTATION)

#  ifdef __CUDACC__
#    define CUDEX_ANNOTATION __host__ __device__
#  else
#    define CUDEX_ANNOTATION
#  endif
#  define CUDEX_ANNOTATION_NEEDS_UNDEF

#elif defined(CUDEX_ANNOTATION_NEEDS_UNDEF)

#undef CUDEX_ANNOTATION
#undef CUDEX_ANNOTATION_NEEDS_UNDEF

#endif

