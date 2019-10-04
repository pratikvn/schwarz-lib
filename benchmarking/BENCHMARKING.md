### Benchmarking.

Example 1.
==========

Poisson solver using Restricted Additive Schwarz with overlap.
-------------------------------------------------------------

The flag `-DSCHWARZ_BUILD_BENCHMARKING` (default `ON`) enables the example and benchmarking snippets.
The following command line options are available for this example. This is setup using `gflags`.

The executable is run in the following fashion:

```sh
[MPI_COMMAND] [MPI_OPTIONS] ${PATH_TO_EXECUTABLE} [FLAGS]
```

Where `[FLAGS]` are the options below with the template [`flag_name [type][default_value]`]. For example, to set the number of iterations of the RAS solver to 100 one would add `--num_iters=100` to the executable command above.

* `num_iters` [uint32][10] : The number of outer iterations for the RAS solver.
* `set_tol` [double][1e-6] : The Outer tolerance for the RAS solver.
* `local_tol` [double][1e-12] : The Inner tolerance for the local iterative solver.
* `set_1d_laplacian_size`[uint32][0] : The number of grid points in one dimension for the 2D laplacian problem.
* `num_refine_cycles` [uint32][1][disabled] : The number of refinement cycles when used with `deal.ii`.
* `enable_onesided` [bool][false] : Enable the onesided asynchronous communication.
* `enable_twosided` [bool][true] : Enable the twosided asynchronous communication. A dummy flag.
* `enable_push_one_by_one` [bool][false][FIXME] : Enable pushing of each element in onesided communication.
* `enable_put_all_local_residual_norms`  [bool][false] : Enable putting of all local residual norms"
* `enable_comm_overlap` [bool][false] : Enable overlap of communication and computation.
* `enable_global_check` [bool][false] : Use the global convergence check for twosided.
* `enable_global_tree_check` [bool][false] : Use the global convergence check for twosided.
* `explicit_laplacian` [bool][false] : Use the explicit laplacian instead of deal.ii's matrix.
* `enable_random_rhs` [bool][false] : Use a random rhs instead of the default 1.0's .
* `overlap` [uint32][2] : Overlap between the domains.
* `executor` [std::string][reference] : The executor used to run the solver, one of `reference`, `cuda` or `omp`.
* `enable_flush` [std::string][flush_all] : The window flush. The choices are `flush_local` and `flush_all`.
* `timings_file` [std::string][null] : The filename for the timings.
* `partition` [std::string][naive] : The partitioner used. The choices are `metis` or `naive`.
* `local_solver` [std::string][direct_cholmod] : The local solver used in the local domains. The current choices are `direct_cholmod` , `direct_ginkgo` or `iterative_ginkgo`.
* `num_threads` [uint32][1], "Number of threads to bind to a process.
* `factor_ordering_natural` [bool][false]: If true uses natural ordering instead of the default optimized ordering. This is needed for CUDA runs as the factorization ordering needs to be given to the solver.

