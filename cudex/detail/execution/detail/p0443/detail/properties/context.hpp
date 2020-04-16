#pragma once

#include "../prologue.hpp"

#if __has_include(<any>)
#include <any>
#endif

#include "detail/basic_executor_property.hpp"


P0443_NAMESPACE_OPEN_BRACE


struct context_t : 
  detail::basic_executor_property<
    context_t,
    false,
    false
#if __cpp_lib_any
    , std::any
#endif
>
{};


static constexpr context_t context{};


P0443_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

