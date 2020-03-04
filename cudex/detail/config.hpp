// note that this header file is special and does not use #pragma once

// This header #includes all header files underneath the detail/config directory.
// The only headers that should #include this file are detail/prologue.hpp and detail/epilogue.hpp
//
// A simple way to redefine configuration macros like CUDEX_NAMESPACE et al is to replace this
// header file with one containing custom definitions for all macros defined beneath the
// detail/config directory.

#include "config/annotation.hpp"
#include "config/exec_check_disable.hpp"
#include "config/has_cudart.hpp"
#include "config/has_exceptions.hpp"
#include "config/namespace.hpp"
#include "config/requires.hpp"

