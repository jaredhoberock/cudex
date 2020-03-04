#pragma once

#include "prologue.hpp"

#include <exception>

P0443_NAMESPACE_OPEN_BRACE


class sink_receiver
{
  public:
    template<class... Args>
    P0443_ANNOTATION
    void set_value(Args&&...) {}

    template<class Arg>
    void set_error(Arg&&) noexcept
    {
      std::terminate();
    }

    void set_done() noexcept {}
};


P0443_NAMESPACE_CLOSE_BRACE

#include "epilogue.hpp"

