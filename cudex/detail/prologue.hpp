// note that this header file is special and does not use #pragma once

#ifndef CUDEX_INCLUDE_LEVEL

// the first time this header is #included, this branch is processed

// this definition communicates that the stack is empty
// and that these macros should be undefined by epilogue.hpp
#define CUDEX_INCLUDE_LEVEL 0

// allow importers of this library to provide a special header to
// be included before the prologue
#if __has_include("foreword.hpp")
#include "foreword.hpp"
#endif

// include preprocessor headers
#include "preprocessor.hpp"

#else

// any other time this header is #included, this branch is processed

// this push to the stack communicates with epilogue.hpp
// that these macros are not ready to be undefined.
#pragma push_macro("CUDEX_INCLUDE_LEVEL")
#undef CUDEX_INCLUDE_LEVEL
#define CUDEX_INCLUDE_LEVEL 1

#endif

