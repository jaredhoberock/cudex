#pragma once

#include "prologue.hpp"

#include <cstdio>
#include <exception>

CUDEX_NAMESPACE_OPEN_BRACE

namespace detail
{
namespace throw_on_error_detail
{


CUDEX_ANNOTATION
inline void print_error_message(cudaError_t e, const char* message) noexcept
{
#if CUDEX_HAS_CUDART
  printf("Error after %s: %s\n", message, cudaGetErrorString(e));
#else
  printf("Error: %s\n", message);
#endif
}


CUDEX_ANNOTATION
inline void terminate() noexcept
{
#ifdef __CUDA_ARCH__
  asm("trap;");
#else
  std::terminate();
#endif
}


} // end throw_on_error_detail


CUDEX_ANNOTATION
inline void throw_on_error(cudaError_t e, const char* message)
{
  if(e)
  {
#ifndef __CUDA_ARCH__
    std::string what = std::string(message) + std::string(": ") + cudaGetErrorString(e);
    throw std::runtime_error(what);
#else
    detail::throw_on_error_detail::print_error_message(e, message);
    detail::throw_on_error_detail::terminate();
#endif
  }
}


} // end detail

CUDEX_NAMESPACE_CLOSE_BRACE

#include "epilogue.hpp"

