// note that this header file is special and does not use #pragma once

// CUDEX_EXEC_CHECK_DISABLE expands to a pragma which tells a CUDA-capable
// compiler not to enforce that a function must call another function with
// a compatible execution space

#ifndef CUDEX_EXEC_CHECK_DISABLE
#  if defined(__CUDACC__) and !defined(__NVCOMPILER_CUDA__)
#    define CUDEX_EXEC_CHECK_DISABLE #pragma nv_exec_check_disable
#  else
#    define CUDEX_EXEC_CHECK_DISABLE
#  endif
#elif defined(CUDEX_EXEC_CHECK_DISABLE)
#  undef CUDEX_EXEC_CHECK_DISABLE
#endif

