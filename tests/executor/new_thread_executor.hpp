#include <cassert>
#include <future>
#include <cudex/executor/new_thread_executor.hpp>
#include <cudex/property/blocking.hpp>


namespace ns = cudex;


void test()
{
  ns::new_thread_executor ex1;

  std::promise<int> p;
  auto fut = p.get_future();

  int expected = 13;

  ex1.execute([expected, p = std::move(p)] () mutable
  {
    p.set_value(expected);
  });

  assert(expected == fut.get());

  ns::new_thread_executor ex2;

  assert(ex1 == ex2);
  assert(!(ex1 != ex2));

  static_assert(ns::blocking.never == ns::new_thread_executor::query(ns::blocking), "Error");
}

void test_new_thread_executor()
{
  test();
}

