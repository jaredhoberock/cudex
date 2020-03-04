This directory contains headers which define and undefine the various preprocessor symbols this library uses.

These headers are meant to be `#include`ed once by `detail/prologue.hpp` and `detail/epilogue.hpp` each. No other file should `#include` these headers.

As such, headers in this directory should not use `#pragma once` or other kinds of include guards.

The first time a header is `#include`d, it should define the symbols it is responsible for. The second, and final, time a header is `#include`d, it should undefine the symbols it introduced the first time.

