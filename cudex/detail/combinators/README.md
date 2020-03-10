This directory contains the implementations of customizable sender combinators such as `then` and others.

For every combinator `foo`, there is a directory `./foo`.

Inside `./foo` are two headers, `./foo/default_foo.hpp` and `./foo/dispatch_foo.hpp`.

There is no "everything" header `./foo.hpp` because a source file typically needs to include one of these headers.

`dispatch_foo.hpp` contains overloads named `dispatch_foo` and type traits:

  * `dispatch_foo_t` - The type of `dispatch_foo(arg1, args...)`'s result.
  * `can_dispatch_foo` - Whether or not the call `dispatch_foo(arg1, args...)` is well-formed.

Unless otherwise specified, `dispatch_foo(arg1, args...)`'s algorithm proceeds as follows:

  1. If `arg1.foo(args...)` is available, return its result.
  2. Else, if `foo(arg1, args...)` is available, return its result.
  3. Else, return `default_foo(arg1, args...)`.

`default_foo.hpp` contains the default implementation of `foo`, named `default_foo`.

