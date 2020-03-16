This directory contains unit tests exercising functionality defined beneath the `cudex` directory.

For example, To build and execute a single test program named `foo`:

    $ make test.foo

To build and execute all test programs:

    $ make test

# Details

Each `<name>.hpp` file in this directory is a C++ file containing a function named `test_<name>`.

A unit test program should define a `main` function which calls the function `test_<name>`. If `test_<name>` exits normally, the test program should print "OK" to stdout, and exit normally.

For example, a unit test named `foo.cpp` should look like:

    #include <iostream>
    #include "foo.hpp"

    int main()
    {
      test_foo();
      std::cout << "OK" << std::endl;
      return 0;
    }

The `Makefile` automates this. To create a new unit test for `foo`, create a header file named `foo.hpp` and define a function named `test_foo`. The `Makefile` will take care of the rest.

# Style Guide

To the degree possible, these unit test programs should be self-contained. This makes it easy to isolate test failures and produce minimal reproducer programs for bug reports.

The tradeoff is that common constructs these programs use are repeated.

