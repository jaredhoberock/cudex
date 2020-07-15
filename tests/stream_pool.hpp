#include <cassert>
#include <cudex/stream_pool.hpp>


namespace ns = cudex;


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


void test_execute(ns::static_stream_pool::executor_type ex, int expected)
{
  ex.execute([=] __device__
  {
    result = expected;
  });
}


__host__ __device__
unsigned int hash_index(ns::kernel_executor::index_type idx)
{
  return idx.block.x ^ idx.block.y ^ idx.block.z ^ idx.thread.x ^ idx.thread.y ^ idx.thread.z;
}


// this array has blockIdx X threadIdx axes
// put 4 elements in each axis
__managed__ unsigned int bulk_result[4][4][4][4][4][4] = {};


void test_bulk_execute(ns::static_stream_pool::executor_type ex)
{
  ns::static_stream_pool::executor_type::shape_type shape{::dim3(4,4,4), ::dim3(4,4,4)};

  ex.bulk_execute([=] __device__ (ns::static_stream_pool::executor_type::index_type idx)
  {
    unsigned int result = hash_index(idx);

    bulk_result[idx.block.x][idx.block.y][idx.block.z][idx.thread.x][idx.thread.y][idx.thread.z] = result;
  }, shape);

  assert(cudaStreamSynchronize(ex.stream()) == cudaSuccess);

  for(unsigned int bx = 0; bx != shape.grid.x; ++bx)
  {
    for(unsigned int by = 0; by != shape.grid.y; ++by)
    {
      for(unsigned int bz = 0; bz != shape.grid.z; ++bz)
      {
        for(unsigned int tx = 0; tx != shape.block.x; ++tx)
        {
          for(unsigned int ty = 0; ty != shape.block.y; ++ty)
          {
            for(unsigned int tz = 0; tz != shape.block.z; ++tz)
            {
              ns::static_stream_pool::executor_type::index_type idx{{bx,by,bz}, {tx,ty,tz}};
              unsigned int expected = hash_index(idx);

              assert(expected == bulk_result[idx.block.x][idx.block.y][idx.block.z][idx.thread.x][idx.thread.y][idx.thread.z]);
            }
          }
        }
      }
    }
  }
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


#ifdef __CUDACC__
  {
    // static_stream_pool::executor_type::bulk_execute requires a CUDA C++ compiler

    static_stream_pool pool(0, 4);

    static_stream_pool::executor_type ex = pool.executor();

    test_bulk_execute(ex);
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

