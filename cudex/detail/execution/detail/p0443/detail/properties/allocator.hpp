#pragma once

#include "../prologue.hpp"

#if __has_include(<any>)
#include <any>
#endif

#include "detail/basic_executor_property.hpp"


P0443_NAMESPACE_OPEN_BRACE


template<class ProtoAllocator>
class allocator_t : 
  detail::basic_executor_property<
    allocator_t<ProtoAllocator>,
    true,
    true
  >
{
  public:
    P0443_ANNOTATION
    constexpr explicit allocator_t(const ProtoAllocator& alloc) : alloc_(alloc) {}

    P0443_ANNOTATION
    constexpr ProtoAllocator value() const
    {
      return alloc_;
    }

  private:
    ProtoAllocator alloc_;
};


template<>
struct allocator_t<void> :
  detail::basic_executor_property<
    allocator_t<void>,
    true,
    true
  >
{
  template<class ProtoAllocator>
  P0443_ANNOTATION
  constexpr allocator_t<ProtoAllocator> operator()(const ProtoAllocator& alloc) const
  {
    return allocator_t<ProtoAllocator>{alloc};
  }
};


static constexpr allocator_t<void> allocator{};


P0443_NAMESPACE_CLOSE_BRACE

#include "../epilogue.hpp"

