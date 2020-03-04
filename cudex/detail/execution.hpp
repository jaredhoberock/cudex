#pragma once

#include "prologue.hpp"

#define P0443_NAMESPACE cudex::detail::execution
#define P0443_NAMESPACE_OPEN_BRACE namespace cudex { namespace detail { namespace execution {
#define P0443_NAMESPACE_CLOSE_BRACE } } }

#include "execution/detail/p0443/execution.hpp"

#undef P0443_NAMESPACE
#undef P0443_NAMESPACE_OPEN_BRACE
#undef P0443_NAMESPACE_CLOSE_BRACE


CUDEX_NAMESPACE_OPEN_BRACE


namespace detail
{
namespace execution
{


// bring the extensions into this namespace
using namespace CUDEX_NAMESPACE::detail::execution::ext;


} // end execution
} // end detail


CUDEX_NAMESPACE_CLOSE_BRACE

#include "epilogue.hpp"

