#pragma once

#include "../detail/prologue.hpp"

#include <cuda_runtime_api.h>
#include <type_traits>
#include <utility>
#include "../detail/throw_on_error.hpp"


CUDEX_NAMESPACE_OPEN_BRACE


class callback_executor
{
  private:
    cudaStream_t stream_;

    template<class T>
    struct self_destructing
    {
      template<class U>
      self_destructing(U&& v)
        : value_{}
      {
        new(&value()) T{std::forward<U>(v)};
      }

      ~self_destructing()
      {
        value().~T();

        std::free(this);
      }

      T& value()
      {
        return *reinterpret_cast<T*>(&value_);
      }

      typename std::aligned_storage<sizeof(T)>::type value_;
    };


    template<class Function>
    static void callback(cudaStream_t stream, cudaError_t status, void* data)
    {
      self_destructing<Function>* ptr_to_f = reinterpret_cast<self_destructing<Function>*>(data);

      if(status == cudaSuccess)
      {
        // call the function
        ptr_to_f->value()();
      }
      else
      {
        // report the error somehow
        // ...
      }

      // call the destructor
      ptr_to_f->~self_destructing<Function>();
    }

  public:
    inline explicit callback_executor(cudaStream_t stream)
      : stream_{stream}
    {}

    inline callback_executor()
      : callback_executor(cudaStream_t{})
    {}

    callback_executor(const callback_executor&) = default;

    template<class Function>
    inline void execute(Function&& f) const noexcept
    {
      using T = typename std::decay<Function>::type;

      // XXX in practice, we should put the callback state inside a stream_pool
      //     and let the stream_pool manage its lifetime
      self_destructing<T> *ptr_to_f = reinterpret_cast<self_destructing<T>*>(std::malloc(sizeof(self_destructing<T>)));
      new(ptr_to_f) self_destructing<T>{std::forward<Function>(f)};

      // enqueue the callback
      detail::throw_on_error(cudaStreamAddCallback(stream_, &callback<T>, ptr_to_f, 0), "stream_callback_executor::execute: CUDA error after cudaStreamAddCallback");
    }

    inline bool operator==(const callback_executor& other) const noexcept
    {
      return stream_ == other.stream_;
    }

    inline bool operator!=(const callback_executor& other) const noexcept
    {
      return !(*this == other);
    }

    inline cudaStream_t stream() const noexcept
    {
      return stream_;
    }
};

CUDEX_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

