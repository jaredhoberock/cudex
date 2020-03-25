#include <cassert>
#include <cudex/stream_pool.hpp>


#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#ifndef __managed__
#define __managed__
#endif

#ifndef __global__
#define __global__
#endif


__managed__ int result;


void test_execute(cudex::static_stream_pool::executor_type ex, int expected)
{
  ex.execute([=] __device__
  {
    result = expected;
  });
}


void test_stream_pool()
{
  using namespace cudex;

#ifdef __CUDACC__
  {
    // static_stream_pool::executor_type::execute requires a CUDA C++ compiler

    static_stream_pool pool(0, 4);

    static_stream_pool::executor_type ex = pool.executor();

    result = 0;
    int expected = 13;

    test_execute(ex, expected);

    pool.wait();

    assert(result == expected);
  }
#endif


  {
    // everything else works in CUDA C++ or C++

    int num_streams = 4;

    static_stream_pool pool(0, num_streams);

    // round-robin through executors
    std::vector<static_stream_pool::executor_type> first_round;
    for(int i = 0; i < num_streams; ++i)
    {
      first_round.push_back(pool.executor());
    }

    std::vector<static_stream_pool::executor_type> second_round;
    for(int i = 0; i < num_streams; ++i)
    {
      second_round.push_back(pool.executor());
    }

    // each executor must equal itself
    for(int i = 0; i < num_streams; ++i)
    {
      assert(first_round[i] == first_round[i]);
      assert(second_round[i] == second_round[i]);
    }

    // each executor in the first round must equal the corresponding one in the second round
    assert(std::equal(first_round.begin(), first_round.end(), second_round.begin()));

    // each executor in each round must not equal other executors in its round
    for(int i = 0; i < num_streams; ++i)
    {
      for(int j = 0; j < num_streams; ++j)
      {
        if(i != j)
        {
          assert(first_round[i] != first_round[j]);
          assert(second_round[i] != second_round[j]);
        }
      }
    }
  }
}

