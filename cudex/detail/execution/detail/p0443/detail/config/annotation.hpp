// P0443_ANNOTATION expands to __host__ __device__ when encountered by a
// CUDA-capable compiler

#if !defined(P0443_ANNOTATION)

#  ifdef __CUDACC__
#    define P0443_ANNOTATION __host__ __device__
#  else
#    define P0443_ANNOTATION
#  endif
#  define P0443_ANNOTATION_NEEDS_UNDEF

#elif defined(P0443_ANNOTATION_NEEDS_UNDEF)

#undef P0443_ANNOTATION
#undef P0443_ANNOTATION_NEEDS_UNDEF

#endif

