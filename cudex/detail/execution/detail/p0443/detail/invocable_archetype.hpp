#pragma once

#include "prologue.hpp"

P0443_NAMESPACE_OPEN_BRACE


struct invocable_archetype
{
  template<class... Args>
  P0443_ANNOTATION
  void operator()(Args&&...) const;
};


P0443_NAMESPACE_CLOSE_BRACE

#include "epilogue.hpp"

