# cudex
CUDA executors

This repository is work-in-progress of a prototype implementation of a executors-based programming model for CUDA C++.

Three concepts organize the model:

* *executors*, representing resources for executing tasks,
* *senders*, representing executable tasks which produce data, and
* *receivers*, representing continuations which consume the data produced by a task's execution.

# Example

In code, these concepts combine to represent the execution of an entire CUDA program:

    // get some data from somewhere
    matrix<float> h_mat = ...
    vector<float> h_vec = ...
 
    // get a GPU executor 
    gpu_executor gpu = cudex::device(0);
 
    // copy data to the gpu
    auto d_mat = just_on(h_mat, gpu);
    auto d_vec = just_on(h_vec, gpu);
 
    // build up some arbitrary task
    auto gpu_task = multiply(d_mat, d_vec)  // do some linear algebra
      .transform(round)                     // then round each element of the resulting vector to the nearest integer
      .sort()                               // then sort the vector
      .unique()                             // then ensure all elements are unique
      .take(100)                            // then discard all but the 100 smallest elements
      .sum()                                // and finally take their sum
    ;

    // submit the gpu task to a receiver that prints the result
    submit(gpu_task, [](int result)
    {
      std::cout << "Received " << result << std::endl;
    });

