// note that this header file is special and does not use #pragma once

#if CUDEX_INCLUDE_LEVEL == 1

// any time this header is #included in a nested fashion, this branch is processed

// pop from the stack
#pragma pop_macro("CUDEX_INCLUDE_LEVEL")

#else

// the final time this header is processed, this branch is taken

#undef CUDEX_INCLUDE_LEVEL

// include preprocessor headers

#include "preprocessor.hpp"

// allow importers of this library to provide a special header to
// be included after the epilogue
#if __has_include("afterword.hpp")
#include "afterword.hpp"
#endif

#endif

