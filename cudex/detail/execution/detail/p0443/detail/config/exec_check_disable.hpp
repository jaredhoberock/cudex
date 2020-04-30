// P0443_EXEC_CHECK_DISABLE expands to a pragma which tells a CUDA-capable
// compiler not to enforce that a function must call another function with
// a compatible execution space

#if !defined(P0443_EXEC_CHECK_DISABLE)
#  if defined(__CUDACC__) and !defined(__NVCOMPILER_CUDA__)
#    define P0443_EXEC_CHECK_DISABLE #pragma nv_exec_check_disable
#  else
#    define P0443_EXEC_CHECK_DISABLE
#  endif
#elif defined(P0443_EXEC_CHECK_DISABLE)
#  undef P0443_EXEC_CHECK_DISABLE
#endif

