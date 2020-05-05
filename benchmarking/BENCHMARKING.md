Benchmarking.                            {#benchmarking_schwarz}
----------------

# Benchmark example 1.
======================

## Poisson solver using Restricted Additive Schwarz with overlap.
----------------------------------------------------------------

The flag `-DSCHWARZ_BUILD_BENCHMARKING` (default `ON`) enables the example and benchmarking snippets.
The following command line options are available for this example. This is setup using `gflags`.

The executable is run in the following fashion:

```sh
[MPI_COMMAND] [MPI_OPTIONS] ${PATH_TO_EXECUTABLE} [FLAGS]
```

Where `[FLAGS]` are the options below with the template [`flag_name [type][default_value]`]. For example, to set the number of iterations of the RAS solver to 100 one would add `--num_iters=100` to the executable command above.

* `num_iters` [uint32][100] : The number of outer iterations for the RAS solver.
* `set_tol` [double][1e-6] : The Outer tolerance for the RAS solver.
* `local_tol` [double][1e-12] : The Inner tolerance for the local iterative solver.
* `set_1d_laplacian_size`[uint32][16] : The number of grid points in one dimension for the 2D laplacian problem.
* `num_refine_cycles` [uint32][1][disabled] : The number of refinement cycles when used with `deal.ii`.
* `enable_onesided` [bool][false] : Enable the onesided asynchronous communication.
* `enable_twosided` [bool][true] : Enable the twosided asynchronous communication. A dummy flag.
* `enable_one_by_one` [bool][false] : Enable putting/getting of each element in onesided communication.
* `enable_put_all_local_residual_norms`  [bool][false] : Enable putting of all local residual norms"
* `enable_comm_overlap` [bool][false] : Enable overlap of communication and computation.
* `enable_global_check` [bool][false] : Use the global convergence check for twosided.
* `global_convergence_type` [std::string][centralized-tree] : Choose the convergence detection algorithm for onesided.
* `enable_decentralized_accumulate` [bool][false] : Use accumulate strategy for decentralized convergence check..
* `explicit_laplacian` [bool][false] : Use the explicit laplacian instead of deal.ii's matrix.
* `rhs_type` [bool][false] : Use a random rhs instead of the default 1.0's .
* `overlap` [uint32][2] : Overlap between the domains.
* `executor` [std::string][reference] : The executor used to run the solver, one of `reference`, `cuda` or `omp`.
* `flush_type` [std::string][flush-all] : The window flush strategy. The choices are `flush-local` and `flush-all`.
* `lock_type` [std::string][lock-all] : The window lock strategy. The choices are `lock-local` and `lock-all`.
* `timings_file` [std::string][null] : The filename for the timings.
* `partition` [std::string][regular] : The partitioner used. The choices are `metis`, `regular` or `regular2d`.
* `metis_objtype` [std::string][null] : The objective type to minimize for the metis partitioner. The choices are `edgecut` and `totalvol`.
* `local_solver` [std::string][iterative-ginkgo] : The local solver used in the local domains. The current choices are `direct-cholmod` , `direct-ginkgo` or `iterative-ginkgo`.
* `num_threads` [uint32][1] : Number of threads to bind to a process.
* `factor_ordering_natural` [bool][false] : If true uses natural ordering instead of the default optimized ordering. This is needed for CUDA runs as the factorization ordering needs to be given to the solver.
* `enable_local_precond` [bool][false] : If true uses the Block jacobi preconditioning for the local iterative solver. 
* `precond_max_block_size` [uint32][16]:  Maximum size of the blocks for the block jacobi preconditioner
* `enable_debug_write` [bool][false] : Enable some debugging outputs to stdout.
* `write_comm_data` [bool][false] : Write the number of sends and recvs of each subdomain to files.
* `print_config` [bool][true] : Print the configuration of the run.
* `remote_comm_type` [std::string][get] : The type of the remote communication. `get` uses `MPI_Get` and `put` uses `MPI_Put`.
* `shifted_iter` [uint32][1] : The number of iterations to communicate for the local subdomains.
